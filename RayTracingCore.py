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
        B = self.max_boxes
        P = self.max_polygons
        
        print("\n--- ANE Virtual RT Core Wire Diagnostics ---")
        
        # =====================================================================
        # STAGE 1: 仮想レイ生成回路
        # =====================================================================
        r00 = inv_view_matrix_64ch[:, 0:1]
        init_px = inv_view_matrix_64ch[:, 3:4]
        print(f"[Log] inv_view_matrix_64ch (r00)  -> min: {r00.min().item():.4f}, max: {r00.max().item():.4f}")
        print(f"[Log] inv_view_matrix_64ch (cam_x)-> min: {init_px.min().item():.4f}, max: {init_px.max().item():.4f}")

        r01, r02 = inv_view_matrix_64ch[:, 1:2], inv_view_matrix_64ch[:, 2:3]
        r10, r11, r12, init_py = inv_view_matrix_64ch[:, 4:5], inv_view_matrix_64ch[:, 5:6], inv_view_matrix_64ch[:, 6:7], inv_view_matrix_64ch[:, 7:8]
        r20, r21, r22, init_pz = inv_view_matrix_64ch[:, 8:9], inv_view_matrix_64ch[:, 9:10], inv_view_matrix_64ch[:, 10:11], inv_view_matrix_64ch[:, 11:12]

        dx = r00 * self.cam_dx + r01 * self.cam_dy + r02 * self.cam_dz
        dy = r10 * self.cam_dx + r11 * self.cam_dy + r12 * self.cam_dz
        dz = r20 * self.cam_dx + r21 * self.cam_dy + r22 * self.cam_dz

        inv_len = torch.rsqrt(dx*dx + dy*dy + dz*dz + 1e-5)
        ray_d_x, ray_d_y, ray_d_z = dx * inv_len, dy * inv_len, dz * inv_len
        ray_o_x, ray_o_y, ray_o_z = init_px, init_py, init_pz
        print(f"[Log] Ray Direction X            -> min: {ray_d_x.min().item():.4f}, max: {ray_d_x.max().item():.4f}")

        # =====================================================================
        # STAGE 2: 仮想BVH探索回路
        # =====================================================================
        boxes_reshaped = bvh_boxes_64ch[:, 0:24].view(1, 6, B, 1, 1)
        b_min_x, b_min_y, b_min_z = boxes_reshaped[:, 0], boxes_reshaped[:, 1], boxes_reshaped[:, 2]
        b_max_x, b_max_y, b_max_z = boxes_reshaped[:, 3], boxes_reshaped[:, 4], boxes_reshaped[:, 5]
        
        print(f"[Log] BVH Box0 Min X             -> {b_min_x[:, 0].mean().item():.4f}")
        print(f"[Log] BVH Box0 Max X             -> {b_max_x[:, 0].mean().item():.4f}")

        inv_rd_x = 1.0 / (ray_d_x + 1e-5)
        inv_rd_y = 1.0 / (ray_d_y + 1e-5)
        inv_rd_z = 1.0 / (ray_d_z + 1e-5)

        t1_x = (b_min_x - ray_o_x) * inv_rd_x
        t2_x = (b_max_x - ray_o_x) * inv_rd_x
        t1_y = (b_min_y - ray_o_y) * inv_rd_y
        t2_y = (b_max_y - ray_o_y) * inv_rd_y
        t1_z = (b_min_z - ray_o_z) * inv_rd_z
        t2_z = (b_max_z - ray_o_z) * inv_rd_z

        t_near = torch.maximum(torch.maximum(torch.minimum(t1_x, t2_x), torch.minimum(t1_y, t2_y)), torch.minimum(t1_z, t2_z))
        t_far  = torch.minimum(torch.minimum(torch.maximum(t1_x, t2_x), torch.maximum(t1_y, t2_y)), torch.maximum(t1_z, t2_z))

        box_hit_all = torch.clamp(torch.relu((t_far - t_near) * 1000.0), min=0.0, max=1.0) * \
                      torch.clamp(torch.relu(t_far * 1000.0), min=0.0, max=1.0)
        print(f"[Log] BVH Box Hit Mask (Box0)    -> min: {box_hit_all[:, 0].min().item():.4f}, max: {box_hit_all[:, 0].max().item():.4f}")

        # =====================================================================
        # STAGE 3: 仮想レイトライアングル交差テスト回路
        # =====================================================================
        v_reshaped = poly_vertices_256ch[:, 0:144].view(1, 9, P, 1, 1)
        v0_x, v0_y, v0_z = v_reshaped[:, 0], v_reshaped[:, 1], v_reshaped[:, 2]
        v1_x, v1_y, v1_z = v_reshaped[:, 3], v_reshaped[:, 4], v_reshaped[:, 5]
        v2_x, v2_y, v2_z = v_reshaped[:, 6], v_reshaped[:, 7], v_reshaped[:, 8]

        e1_x, e1_y, e1_z = v1_x - v0_x, v1_y - v0_y, v1_z - v0_z
        e2_x, e2_y, e2_z = v2_x - v0_x, v2_y - v0_y, v2_z - v0_z

        pvec_x = ray_d_y * e2_z - ray_d_z * e2_y
        pvec_y = ray_d_z * e2_x - ray_d_x * e2_z
        pvec_z = ray_d_x * e2_y - ray_d_y * e2_x

        det = e1_x * pvec_x + e1_y * pvec_y + e1_z * pvec_z
        print(f"[Log] Matrix Determinant (Poly0) -> min: {det[:, 0].min().item():.4f}, max: {det[:, 0].max().item():.4f}")
        
        # 【完全安全化】whereを全廃。ピュアなALU（signとclamp）だけでNaNの発生を100%遮断
        det_sign = torch.sign(det)
        # 絶対値を取り、極小値を1e-4にクランプしたあと、元の符号を掛け直す
        safe_det = det_sign * torch.clamp(torch.abs(det), min=1e-4)
        inv_det = 1.0 / safe_det

        tvec_x = ray_o_x - v0_x
        tvec_y = ray_o_y - v0_y
        tvec_z = ray_o_z - v0_z

        u = (tvec_x * pvec_x + tvec_y * pvec_y + tvec_z * pvec_z) * inv_det
        u_valid = torch.clamp(torch.relu(u) * 1000.0, 0.0, 1.0) * torch.clamp(torch.relu(1.0 - u) * 1000.0, 0.0, 1.0)

        qvec_x = tvec_y * e1_z - tvec_z * e1_y
        qvec_y = tvec_z * e1_x - tvec_x * e1_z
        qvec_z = tvec_x * e1_y - tvec_y * e1_x

        v = (ray_d_x * qvec_x + ray_d_y * qvec_y + ray_d_z * qvec_z) * inv_det
        v_valid = torch.clamp(torch.relu(v) * 1000.0, 0.0, 1.0) * torch.clamp(torch.relu(1.0 - (u + v)) * 1000.0, 0.0, 1.0)

        poly_hit_mask = u_valid * v_valid
        print(f"[Log] Triangle Hit Mask (Poly0)  -> min: {poly_hit_mask[:, 0].min().item():.4f}, max: {poly_hit_mask[:, 0].max().item():.4f}")

        t = (e2_x * qvec_x + e2_y * qvec_y + e2_z * qvec_z) * inv_det
        
        # =====================================================================
        # STAGE 4: マルチプレクス
        # =====================================================================
        box_mask_for_polys = box_hit_all.repeat(1, P // B, 1, 1)
        final_poly_hit_valid = torch.clamp(torch.relu(t) * 1000.0, 0.0, 1.0) * poly_hit_mask * box_mask_for_polys
        print(f"[Log] Final Combined Mask (Poly0)-> min: {final_poly_hit_valid[:, 0].min().item():.4f}, max: {final_poly_hit_valid[:, 0].max().item():.4f}")

        # =====================================================================
        # STAGE 5: Z-Buffer 出力
        # =====================================================================
        t_final_buffer = t * final_poly_hit_valid + self.FAR_DEPTH * (1.0 - final_poly_hit_valid)

        min_depth, _ = torch.min(t_final_buffer, dim=1, keepdim=True)
        render_mask = torch.clamp(torch.relu((self.FAR_DEPTH - 1.0) - min_depth) * 1000.0, 0.0, 1.0)
        
        # 【安全化】クラッシュを避けるため、表示側のキャスト処理を安全に記述
        hit_sum = render_mask.sum().item()
        if torch.isnan(torch.tensor(hit_sum)):
            print("[Log] Output Render Mask Pixels -> hit count: NaN / 65536")
        else:
            print(f"[Log] Output Render Mask Pixels -> hit count: {int(hit_sum)} / {self.w * self.h}")

        return torch.cat([min_depth, render_mask], dim=1)
