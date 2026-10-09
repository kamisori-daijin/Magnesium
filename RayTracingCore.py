import torch
import torch.nn as nn

class ANEVirtualRTCore(nn.Module):
    def __init__(self, width=256, height=256, max_boxes=4, max_polygons=16):
        super().__init__()
        self.w = width
        self.h = height
        self.max_boxes = max_boxes
        self.max_polygons = max_polygons
        
        # 1. カメラレイ生成用の座標グリッド (最小形状で自動ブロードキャスト)
        y_grid = torch.linspace(1.0, -1.0, self.h).view(1, 1, self.h, 1)
        x_grid = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        self.register_buffer("cam_dx", x_grid.half())
        self.register_buffer("cam_dy", y_grid.half())
        self.register_buffer("cam_dz", torch.tensor([[[[-1.0]]]]).half())
        
        # 2. 回路定数 (無限遠の初期深度バッファ)
        self.register_buffer("FAR_DEPTH", torch.tensor([[[[10000.0]]]]).half())

    def forward(self, inv_view_matrix_64ch, bvh_boxes_64ch, poly_vertices_256ch):
        """
        ANEの物理特性(64chアライメント)に完全に準拠した、仮想RTコアの物理回路シミュレータ
        Args:
            inv_view_matrix_64ch: [1, 64, 1, 1]  (アライメント済。0~11ch: カメラ3x4行列, 12~63ch: 拡張用無料枠)
            bvh_boxes_64ch:       [1, 64, 1, 1]  (アライメント済。0~23ch: Max4個の箱のMin/Max座標, 24~63ch: パディング)
            poly_vertices_256ch:  [1, 256, 1, 1] (アライメント済。0~143ch: Max16個のポリゴン頂点, 144~255ch: パディング)
        """
        B = self.max_boxes
        P = self.max_polygons
        
        # =====================================================================
        # STAGE 1: 仮想レイ生成回路 (64chの太い束から連続直球読み込み)
        # =====================================================================
        # 64chの塊のまま入力されているため、不連続なギャザーを起こさず最速でALUへ流れます
        r00 = inv_view_matrix_64ch[:, 0:1]
        r01 = inv_view_matrix_64ch[:, 1:2]
        r02 = inv_view_matrix_64ch[:, 2:3]
        init_px = inv_view_matrix_64ch[:, 3:4]
        
        r10 = inv_view_matrix_64ch[:, 4:5]
        r11 = inv_view_matrix_64ch[:, 5:6]
        r12 = inv_view_matrix_64ch[:, 6:7]
        init_py = inv_view_matrix_64ch[:, 7:8]
        
        r20 = inv_view_matrix_64ch[:, 8:9]
        r21 = inv_view_matrix_64ch[:, 9:10]
        r22 = inv_view_matrix_64ch[:, 10:11]
        init_pz = inv_view_matrix_64ch[:, 11:12]

        # 画面の全ピクセル([1,1,H,W])に、カメラパラメータが一斉にブロードキャスト掛け算(Elementwise ALU)される
        dx = r00 * self.cam_dx + r01 * self.cam_dy + r02 * self.cam_dz
        dy = r10 * self.cam_dx + r11 * self.cam_dy + r12 * self.cam_dz
        dz = r20 * self.cam_dx + r21 * self.cam_dy + r22 * self.cam_dz

        # ANE専用ハードウェア命令 (Activations: rsqrt)
        inv_len = torch.rsqrt(dx*dx + dy*dy + dz*dz + 1e-5)
        ray_d_x, ray_d_y, ray_d_z = dx * inv_len, dy * inv_len, dz * inv_len
        ray_o_x, ray_o_y, ray_o_z = init_px, init_py, init_pz

        # =====================================================================
        # STAGE 2: 仮想BVH探索回路 (AABB 不等式一括 ALU マスク処理)
        # =====================================================================
        # 有効な24ch（6要素×4箱）だけを抽出し、C軸（並列プロセッサ軸）に展開
        # 残りの40chのパディングは転送単位の調整として背後で自動無視されます
        boxes_reshaped = bvh_boxes_64ch[:, 0:24].view(1, 6, B, 1, 1)
        b_min_x, b_min_y, b_min_z = boxes_reshaped[:, 0], boxes_reshaped[:, 1], boxes_reshaped[:, 2]
        b_max_x, b_max_y, b_max_z = boxes_reshaped[:, 3], boxes_reshaped[:, 4], boxes_reshaped[:, 5]

        # 各面との交差時間を全ピクセル・全箱同時に一括並列計算 (Elementwise ALU)
        inv_rd_x = 1.0 / (ray_d_x + 1e-5)
        inv_rd_y = 1.0 / (ray_d_y + 1e-5)
        inv_rd_z = 1.0 / (ray_d_z + 1e-5)

        t1_x = (b_min_x - ray_o_x) * inv_rd_x
        t2_x = (b_max_x - ray_o_x) * inv_rd_x
        t1_y = (b_min_y - ray_o_y) * inv_rd_y
        t2_y = (b_max_x - ray_o_y) * inv_rd_y
        t1_z = (b_min_z - ray_o_z) * inv_rd_z
        t2_z = (b_max_z - ray_o_z) * inv_rd_z

        # 箱の入り口(t_near)と出口(t_far)を ANE ネイティブの min/max ユニットで判定
        t_min_x = torch.minimum(t1_x, t2_x)
        t_max_x = torch.maximum(t1_x, t2_x)
        t_min_y = torch.minimum(t1_y, t2_y)
        t_max_y = torch.maximum(t1_y, t2_y)
        t_min_z = torch.minimum(t1_z, t2_z)
        t_max_z = torch.maximum(t1_z, t2_z)

        t_near = torch.maximum(torch.maximum(t_min_x, t_min_y), t_min_z)
        t_far  = torch.maximum(torch.minimum(t_max_x, t_max_y), t_max_z)

        # 【条件分岐の大改造】 不等式をビットマスク（ワイヤー電圧）に変換するReLU回路
        box_hit_all = torch.clamp(torch.relu((t_far - t_near) * 1000.0), min=0.0, max=1.0) * \
                      torch.clamp(torch.relu(t_far * 1000.0), min=0.0, max=1.0) # [1, B, H, W]

        # =====================================================================
        # STAGE 3: 仮想レイトライアングル交差テスト回路 (Möller-Trumbore ベクトル化)
        # =====================================================================
        # 有効な144ch（9要素×16ポリゴン）を抽出し、C軸（プロセッサ軸）に展開
        # 144〜255chのパディングはアライメント整合用として完全にスルーされます
        v_reshaped = poly_vertices_256ch[:, 0:144].view(1, 9, P, 1, 1)
        v0_x, v0_y, v0_z = v_reshaped[:, 0], v_reshaped[:, 1], v_reshaped[:, 2]
        v1_x, v1_y, v1_z = v_reshaped[:, 3], v_reshaped[:, 4], v_reshaped[:, 5]
        v2_x, v2_y, v2_z = v_reshaped[:, 6], v_reshaped[:, 7], v_reshaped[:, 8]

        # エッジベクトルの計算
        e1_x, e1_y, e1_z = v1_x - v0_x, v1_y - v0_y, v1_z - v0_z
        e2_x, e2_y, e2_z = v2_x - v0_x, v2_y - v0_y, v2_z - v0_z

        # 外積 (pvec = ray_d × e2)
        pvec_x = ray_d_y * e2_z - ray_d_z * e2_y
        pvec_y = ray_d_z * e2_x - ray_d_x * e2_z
        pvec_z = ray_d_x * e2_y - ray_d_y * e2_x

        # 内積 (det = e1 ⋅ pvec)
        det = e1_x * pvec_x + e1_y * pvec_y + e1_z * pvec_z
        inv_det = 1.0 / (det + 1e-5) # Dedicated Hardware Reciprocal

        # 交点座標へのアプローチ (tvec = ray_o - v0)
        tvec_x = ray_o_x - v0_x
        tvec_y = ray_o_y - v0_y
        tvec_z = ray_o_z - v0_z

        # 重心座標 u の計算と範囲判定マスク
        u = (tvec_x * pvec_x + tvec_y * pvec_y + tvec_z * pvec_z) * inv_det
        u_valid = torch.clamp(torch.relu(u) * 1000.0, 0.0, 1.0) * torch.clamp(torch.relu(1.0 - u) * 1000.0, 0.0, 1.0)

        # 外積 (qvec = tvec × e1)
        qvec_x = tvec_y * e1_z - tvec_z * e1_y
        qvec_y = tvec_z * e1_x - tvec_x * e1_z
        qvec_z = tvec_x * e1_y - tvec_y * e1_x

        # 重心座標 v の計算と範囲判定マスク
        v = (ray_d_x * qvec_x + ray_d_y * qvec_y + ray_d_z * qvec_z) * inv_det
        v_valid = torch.clamp(torch.relu(v) * 1000.0, 0.0, 1.0) * torch.clamp(torch.relu(1.0 - (u + v)) * 1000.0, 0.0, 1.0)

        poly_hit_mask = u_valid * v_valid # 三角形の内側にいるピクセル [1, P, H, W]

        # 衝突深度 t の計算
        t = (e2_x * qvec_x + e2_y * qvec_y + e2_z * qvec_z) * inv_det
        
        # =====================================================================
        # STAGE 4: マルチプレクス（BVHフラグとの物理配線結合）
        # =====================================================================
        # 16個のポリゴンに対して、どの箱（4個）に属しているかの親子配線を固定化
        # [1, 4, H, W] -> repeatで [1, 16, H, W] へ（ノーコストview＆ブロードキャスト）
        box_mask_for_polys = box_hit_all.repeat(1, P // B, 1, 1)
        
        # 箱に当たっていなければ（0.0）、その中にある全ポリゴンのフラグを一瞬で電気的に強制遮断（0.0）
        final_poly_hit_valid = torch.clamp(torch.relu(t) * 1000.0, 0.0, 1.0) * poly_hit_mask * box_mask_for_polys

        # =====================================================================
        # STAGE 5: Z-Buffer 出力（Native Reductions）
        # =====================================================================
        # 当たらなかったポリゴンの深度は無限遠（10000.0）にする
        t_final_buffer = t * final_poly_hit_valid + self.FAR_DEPTH * (1.0 - final_poly_hit_valid)

        # ANEの最強 Reduce (min) ユニットを駆動し、最も手前のポリゴン深度を全ピクセル一瞬で特定
        min_depth, _ = torch.min(t_final_buffer, dim=1, keepdim=True)

        # 背景（何も当たっていないピクセル）をはじく最終描画マスク
        render_mask = torch.clamp(torch.relu((self.FAR_DEPTH - 1.0) - min_depth) * 1000.0, 0.0, 1.0)

        # 5ch一括返しに合わせるため、出力もアライメントを意識して4ch（または64ch）にパッキングして返すのが本来の理想ですが、
        # ここでは描画テストがしやすいように [深度、マスク] の最小限のチャンネル結合でリターンします
        return torch.cat([min_depth, render_mask], dim=1)
