import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class ANERayTracingCore(nn.Module):
    def __init__(self, width=256, height=256, max_steps=64, shadow_steps=16):
        super().__init__()
        self.w = width
        self.h = height
        self.max_steps = max_steps
        self.shadow_steps = shadow_steps
        self.dt = 0.08
        self.eps = 0.02  # 数値微分用の微小変化量
        
        # --- カメラ空間での初期レイ（方向・位置）の生成 ---
        y_grid = torch.linspace(1.0, -1.0, self.h).view(1, 1, self.h, 1)
        x_grid = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        
        self.register_buffer("cam_dx", x_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dy", y_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dz", torch.full((1, 1, self.h, self.w), -1.0).half())
        
        self.floor_y = -0.8
        self.register_buffer("ONES", torch.ones(1, 1, self.h, self.w).half())
        self.register_buffer("ZEROS", torch.zeros(1, 1, self.h, self.w).half())
        
        # 平行光源の方向（ワールド空間）
        light_dir_x = torch.full((1, 1, 1, 1), 1.0)
        light_dir_y = torch.full((1, 1, 1, 1), 1.0)
        light_dir_z = torch.full((1, 1, 1, 1), 1.0)
        inv_l_len = torch.rsqrt(light_dir_x*light_dir_x + light_dir_y*light_dir_y + light_dir_z*light_dir_z + 1e-5)
        self.register_buffer("light_dx", (light_dir_x * inv_l_len).half())
        self.register_buffer("light_dy", (light_dir_y * inv_l_len).half())
        self.register_buffer("light_dz", (light_dir_z * inv_l_len).half())

        # ANE最適化: permuteを全滅させるため、最初からステップ数を【Channel次元 (Dim 1)】に配置
        self.register_buffer("step_ratios", (torch.arange(self.max_steps).view(1, self.max_steps, 1, 1) * self.dt).half())
        self.register_buffer("shadow_ratios", (torch.arange(self.shadow_steps).view(1, self.shadow_steps, 1, 1) * self.dt).half())

        # RTコア化ハック: 3面図マスク金型をモデル内部に完全にアセット焼き込み！
        y_tex = torch.linspace(-1.0, 1.0, self.h).view(1, 1, self.h, 1)
        x_tex = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        base_mask_x = torch.clamp(1.0 - (torch.abs(x_tex) - 0.4) * 100.0, min=0.0, max=1.0)
        base_mask_y = torch.clamp(1.0 - (torch.abs(y_tex) - 0.4) * 100.0, min=0.0, max=1.0)
        cube_2d_mask = (base_mask_x * base_mask_y).half()

        # XY, XZ, YZ としてモデル内部にバッファ登録
        self.register_buffer("base_multiview_textures", torch.cat([cube_2d_mask, cube_2d_mask, cube_2d_mask], dim=1))

        # 🌟 🛠️ ANE専用：torch.cumsum を駆逐する 1x1 畳み込みフィルターの定義
        # [max_steps -> max_steps] への 1x1 Conv を累積和（下三角行列）の重みで初期化
        self.ane_cumsum_conv = nn.Conv2d(
            in_channels=self.max_steps, 
            out_channels=self.max_steps, 
            kernel_size=1, 
            bias=False
        )
        weight_matrix = torch.tril(torch.ones(self.max_steps, self.max_steps))
        self.ane_cumsum_conv.weight.data = weight_matrix.view(self.max_steps, self.max_steps, 1, 1).half()
        self.ane_cumsum_conv.weight.requires_grad = False

    def check_multiview_hit(self, px, py, pz):
        """
        px, py, pz: warp coordinates
        """
        # Box boundary check（オブジェクトのローカル空間の境界判定）
        out_x = torch.relu(torch.abs(px) - 1.0)
        out_y = torch.relu(torch.abs(py) - 1.0)
        out_z = torch.relu(torch.abs(pz) - 1.0)
        any_out = torch.clamp((out_x + out_y + out_z) * 100.0, min=0.0, max=1.0)
        box_check = 1.0 - any_out

        # CPUで行っていた「abs(軸) <= 0.4」をReLUと一斉クランプの数式だけでANE上に完全シミュレート！
        proj_xy = torch.clamp(1.0 - torch.relu(torch.abs(px) - 0.4) * 10.0, 0.0, 1.0) * \
                  torch.clamp(1.0 - torch.relu(torch.abs(py) - 0.4) * 10.0, 0.0, 1.0)
                  
        proj_xz = torch.clamp(1.0 - torch.relu(torch.abs(px) - 0.4) * 10.0, 0.0, 1.0) * \
                  torch.clamp(1.0 - torch.relu(torch.abs(pz) - 0.4) * 10.0, 0.0, 1.0)
                  
        proj_yz = torch.clamp(1.0 - torch.relu(torch.abs(py) - 0.4) * 10.0, 0.0, 1.0) * \
                  torch.clamp(1.0 - torch.relu(torch.abs(pz) - 0.4) * 10.0, 0.0, 1.0)

        # マスクの幾何学結合
        mask_xy = self.base_multiview_textures[:, 0:1, :, :] * proj_xy
        mask_xz = self.base_multiview_textures[:, 1:2, :, :] * proj_xz
        mask_yz = self.base_multiview_textures[:, 2:3, :, :] * proj_yz

        object_hit = mask_xy * mask_xz * mask_yz * box_check
        return object_hit

        
    def forward(self, multiview_textures, inv_view_matrix_64d):
        """
        Input:
              multiview_textures: [1, 3, 256, 256] (ダミー)
              inv_view_matrix_64d: [1, 64, 1, 1] (Swift側から届く統合マトリクス)
        """
        # ==========================================
        # 🗺️ 1. カメラ逆行列 & オブジェクト逆モデル行列(2個分)の固定展開
        # ==========================================
        # 最初の16chからカメラの逆行列を復元
        inv_view = inv_view_matrix_64d[0, :16, 0, 0].view(4, 4).half()

        # 16〜31chからオブジェクト1の「モデル逆行列」を復元
        inv_model1 = inv_view_matrix_64d[0, 16:32, 0, 0].view(4, 4).half()

        # 32〜47chからオブジェクト2の「モデル逆行列」を復元 🌟固定Shape拡張！
        inv_model2 = inv_view_matrix_64d[0, 32:48, 0, 0].view(4, 4).half()

        # カメラ行列の分解によるレイ方向の算出
        r00, r01, r02 = inv_view[0, 0], inv_view[0, 1], inv_view[0, 2]
        r10, r11, r12 = inv_view[1, 0], inv_view[1, 1], inv_view[1, 2]
        r20, r21, r22 = inv_view[2, 0], inv_view[2, 1], inv_view[2, 2]

        dx = r00 * self.cam_dx + r01 * self.cam_dy + r02 * self.cam_dz
        dy = r10 * self.cam_dx + r11 * self.cam_dy + r12 * self.cam_dz
        dz = r20 * self.cam_dx + r21 * self.cam_dy + r22 * self.cam_dz

        inv_len = torch.rsqrt(dx*dx + dy*dy + dz*dz + 1e-5)
        init_dx = dx * inv_len
        init_dy = dy * inv_len
        init_dz = dz * inv_len

        init_px = inv_view[0, 3].view(1, 1, 1, 1)
        init_py = inv_view[1, 3].view(1, 1, 1, 1)
        init_pz = inv_view[2, 3].view(1, 1, 1, 1)

        # ==========================================
        # 🚀 2. メインの視線レイマーチング (🌟完全にChannel次元並列)
        # ==========================================
        # [1, max_steps, H, W] へ一撃ブロードキャスト
        px_all = init_px + init_dx * self.step_ratios
        py_all = init_py + init_dy * self.step_ratios
        pz_all = init_pz + init_dz * self.step_ratios

        # --- オブジェクト1の空間ワープ ---
        m1_00, m1_01, m1_02, m1_03 = inv_model1[0, 0], inv_model1[0, 1], inv_model1[0, 2], inv_model1[0, 3]
        m1_10, m1_11, m1_12, m1_13 = inv_model1[1, 0], inv_model1[1, 1], inv_model1[1, 2], inv_model1[1, 3]
        m1_20, m1_21, m1_22, m1_23 = inv_model1[2, 0], inv_model1[2, 1], inv_model1[2, 2], inv_model1[2, 3]

        local1_px_all = m1_00 * px_all + m1_01 * py_all + m1_02 * pz_all + m1_03
        local1_py_all = m1_10 * px_all + m1_11 * py_all + m1_12 * pz_all + m1_13
        local1_pz_all = m1_20 * px_all + m1_21 * py_all + m1_22 * pz_all + m1_23

        # --- オブジェクト2の空間ワープ 🌟固定並列化 ---
        m2_00, m2_01, m2_02, m2_03 = inv_model2[0, 0], inv_model2[0, 1], inv_model2[0, 2], inv_model2[0, 3]
        m2_10, m2_11, m2_12, m2_13 = inv_model2[1, 0], inv_model2[1, 1], inv_model2[1, 2], inv_model2[1, 3]
        m2_20, m2_21, m2_22, m2_23 = inv_model2[2, 0], inv_model2[2, 1], inv_model2[2, 2], inv_model2[2, 3]

        local2_px_all = m2_00 * px_all + m2_01 * py_all + m2_02 * pz_all + m2_03
        local2_py_all = m2_10 * px_all + m2_11 * py_all + m2_12 * pz_all + m2_13
        local2_pz_all = m2_20 * px_all + m2_21 * py_all + m2_22 * pz_all + m2_23

        # それぞれの金型シルエットと交差判定
        hit1_all = self.check_multiview_hit(local1_px_all, local1_py_all, local1_pz_all)
        hit2_all = self.check_multiview_hit(local2_px_all, local2_py_all, local2_pz_all)

        # 🌟 ANEゴリ押し最大値合成（1つの空間の論理和として結合！）
        object_hit_all = torch.max(hit1_all, hit2_all)

        # 床の不変判定と合算
        floor_hit_all = torch.clamp(torch.relu(self.floor_y - py_all) * 100.0, min=0.0, max=1.0)
        any_hit_all = torch.clamp(object_hit_all + floor_hit_all, min=0.0, max=1.0)

        # 🛠️ 1x1 Conv による完全ANEネイティブ累積和
        cum_hit = self.ane_cumsum_conv(any_hit_all)

        prior_hit = torch.cat([torch.zeros_like(cum_hit[:, :1, :, :]), cum_hit[:, :-1, :, :]], dim=1)
        not_hit_yet_all = torch.clamp(1.0 - prior_hit, min=0.0, max=1.0)

        is_first_object_all = not_hit_yet_all * object_hit_all
        is_first_floor_all = not_hit_yet_all * (1.0 - object_hit_all) * floor_hit_all

        # 各ステップのフラグを総和して縮約
        hit_object_mask = torch.sum(is_first_object_all, dim=1, keepdim=True).clamp(0.0, 1.0)
        hit_floor_mask = torch.sum(is_first_floor_all, dim=1, keepdim=True).clamp(0.0, 1.0)
        accum_hit = torch.clamp(hit_object_mask + hit_floor_mask, min=0.0, max=1.0)

        # 最終衝突したワールド3D座標
        px = torch.sum(is_first_object_all * px_all + is_first_floor_all * px_all, dim=1, keepdim=True)
        py = torch.sum(is_first_object_all * py_all + is_first_floor_all * py_all, dim=1, keepdim=True)
        pz = torch.sum(is_first_object_all * pz_all + is_first_floor_all * pz_all, dim=1, keepdim=True)

        # ==========================================
        # 📐 3. 全自動・数値微分による法線計算（当たった方を微分）
        # ==========================================
        local1_px = m1_00 * px + m1_01 * py + m1_02 * pz + m1_03
        local1_py = m1_10 * px + m1_11 * py + m1_12 * pz + m1_13
        local1_pz = m1_20 * px + m1_21 * py + m1_22 * pz + m1_23

        local2_px = m2_00 * px + m2_01 * py + m2_02 * pz + m2_03
        local2_py = m2_10 * px + m2_11 * py + m2_12 * pz + m2_13
        local2_pz = m2_20 * px + m2_21 * py + m2_22 * pz + m2_23

        # 2つのオブジェクトそれぞれのローカル勾配を抽出
        f1_c = self.check_multiview_hit(local1_px, local1_py, local1_pz)
        f1_x = self.check_multiview_hit(local1_px + self.eps, local1_py, local1_pz)
        f1_y = self.check_multiview_hit(local1_px, local1_py + self.eps, local1_pz)
        f1_z = self.check_multiview_hit(local1_px, local1_py, local1_pz + self.eps)

        f2_c = self.check_multiview_hit(local2_px, local2_py, local2_pz)
        f2_x = self.check_multiview_hit(local2_px + self.eps, local2_py, local2_pz)
        f2_y = self.check_multiview_hit(local2_px, local2_py + self.eps, local2_pz)
        f2_z = self.check_multiview_hit(local2_px, local2_py, local2_pz + self.eps)

        # 🌟 どちらのオブジェクトに衝突したか判定マスクを作り、法線計算をブレンド
        # (if文を排除し、両方計算してマスクで乗算するANEゴリ押しスタイル)
        obj1_mask = torch.sum(is_first_object_all * hit1_all, dim=1, keepdim=True).clamp(0.0, 1.0)
        obj2_mask = torch.sum(is_first_object_all * (1.0 - hit1_all) * hit2_all, dim=1, keepdim=True).clamp(0.0, 1.0)

        raw_nx = obj1_mask * (f1_c - f1_x) + obj2_mask * (f2_c - f2_x)
        raw_ny = obj1_mask * (f1_c - f1_y) + obj2_mask * (f2_c - f2_y)
        raw_nz = obj1_mask * (f1_c - f1_z) + obj2_mask * (f2_c - f2_z)

        # ワールド空間に戻す（回転成分の適用もブレンド）
        w_nx1 = m1_00 * (f1_c - f1_x) + m1_10 * (f1_c - f1_y) + m1_20 * (f1_c - f1_z)
        w_ny1 = m1_01 * (f1_c - f1_x) + m1_11 * (f1_c - f1_y) + m1_21 * (f1_c - f1_z)
        w_nz1 = m1_02 * (f1_c - f1_x) + m1_12 * (f1_c - f1_y) + m1_22 * (f1_c - f1_z)

        w_nx2 = m2_00 * (f2_c - f2_x) + m2_10 * (f2_c - f2_y) + m2_20 * (f2_c - f2_z)
        w_ny2 = m2_01 * (f2_c - f2_x) + m2_11 * (f2_c - f2_y) + m2_21 * (f2_c - f2_z)
        w_nz2 = m2_02 * (f2_c - f2_x) + m2_12 * (f2_c - f2_y) + m2_22 * (f2_c - f2_z)

        world_nx = obj1_mask * w_nx1 + obj2_mask * w_nx2
        world_ny = obj1_mask * w_ny1 + obj2_mask * w_ny2
        world_nz = obj1_mask * w_nz1 + obj2_mask * w_nz2

        inv_true_n_len = torch.rsqrt(world_nx*world_nx + world_ny*world_ny + world_nz*world_nz + 1e-5)
        
        first_nx = hit_object_mask * (world_nx * inv_true_n_len)
        first_ny = hit_object_mask * (world_ny * inv_true_n_len) + hit_floor_mask * 1.0
        first_nz = hit_object_mask * (world_nz * inv_true_n_len)

        # ==========================================
        # 👤 4. シャドウレイ (🌟オブジェクト2個対応)
        # ==========================================
        shadow_start_x = px + 0.04 * self.light_dx
        shadow_start_y = py + 0.04 * self.light_dy
        shadow_start_z = pz + 0.04 * self.light_dz

        spx_all = shadow_start_x + self.light_dx * self.shadow_ratios
        spy_all = shadow_start_y + self.light_dy * self.shadow_ratios
        spz_all = shadow_start_z + self.light_dz * self.shadow_ratios

        # 両方のオブジェクトの影を並列判定
        l1_spx_all = m1_00 * spx_all + m1_01 * spy_all + m1_02 * spz_all + m1_03
        l1_spy_all = m1_10 * spx_all + m1_11 * spy_all + m1_12 * spz_all + m1_13
        l1_spz_all = m1_20 * spx_all + m1_21 * spy_all + m1_22 * spz_all + m1_23

        l2_spx_all = m2_00 * spx_all + m2_01 * spy_all + m2_02 * spz_all + m2_03
        l2_spy_all = m2_10 * spx_all + m2_11 * spy_all + m2_12 * spz_all + m2_13
        l2_spz_all = m2_20 * spx_all + m2_21 * spy_all + m2_22 * spz_all + m2_23

        s_hit1 = self.check_multiview_hit(l1_spx_all, l1_spy_all, l1_spz_all)
        s_hit2 = self.check_multiview_hit(l2_spx_all, l2_spy_all, l2_spz_all)
        shadow_hit_all = torch.max(s_hit1, s_hit2)

        accum_shadow = torch.sum(shadow_hit_all, dim=1, keepdim=True).clamp(0.0, 1.0) * hit_floor_mask

        # ==========================================
        # 🎨 5. ライティング ＆ チェッカー床
        # ==========================================
        diffuse = first_nx * self.light_dx + first_ny * self.light_dy + first_nz * self.light_dz
        shading = torch.relu(diffuse) + 0.15

        sign_x = torch.clamp(px * 3.0 * 100.0, min=-1.0, max=1.0)
        sign_z = torch.clamp(pz * 3.0 * 100.0, min=-1.0, max=1.0)
        checker = (sign_x * sign_z + 1.0) * 0.5
        floor_color = hit_floor_mask * (0.3 + 0.2 * checker)

        # 🌟 おまけハック: オブジェクト1は白(ONES)、オブジェクト2はほんのり青(0.7, 0.8, 1.0)に塗り分け！
        obj1_color = obj1_mask * self.ONES
        obj2_color_r = obj2_mask * 0.7
        obj2_color_g = obj2_mask * 0.8
        obj2_color_b = obj2_mask * 1.0
        
        # カラーバッファの合成 [1, 3, H, W] に対応できるよう拡張
        rgb_object_color = torch.cat([obj1_color + obj2_color_r, obj1_color + obj2_color_g, obj1_color + obj2_color_b], dim=1)
        rgb_floor_color = torch.cat([floor_color, floor_color, floor_color], dim=1)

        base_color = rgb_object_color + rgb_floor_color
        light_modifier = (self.ONES - accum_shadow) * shading + accum_shadow * 0.15

        return accum_hit * base_color * light_modifier
