import torch
import torch.nn as nn

class ANEVirtualRTCore(nn.Module):
    def __init__(self, width=256, height=256, max_boxes=4, max_polygons=16):
        super().__init__()
        self.w = width
        self.h = height
        self.max_boxes = max_boxes
        self.max_polygons = max_polygons
        
        # 1. カメラレイ生成用の座標グリッド
        y_grid = torch.linspace(1.0, -1.0, self.h).view(1, 1, self.h, 1)
        x_grid = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        self.register_buffer("cam_dx", x_grid.half())
        self.register_buffer("cam_dy", y_grid.half())
        self.register_buffer("cam_dz", torch.tensor([[[[-1.0]]]]).half())
        
        # 2. 回路定数
        self.register_buffer("FAR_DEPTH", torch.tensor([[[[10000.0]]]]).half())

    def forward(self, inv_view_matrix_64ch, bvh_boxes_64ch, poly_vertices_256ch):
        B = self.max_boxes
        P = self.max_polygons
        
        # =====================================================================
        # STAGE 1: 仮想レイ生成回路
        # =====================================================================
        r00, r01, r02 = inv_view_matrix_64ch[:, 0:1], inv_view_matrix_64ch[:, 1:2], inv_view_matrix_64ch[:, 2:3]
        r10, r11, r12 = inv_view_matrix_64ch[:, 4:5], inv_view_matrix_64ch[:, 5:6], inv_view_matrix_64ch[:, 6:7]
        r20, r21, r22 = inv_view_matrix_64ch[:, 8:9], inv_view_matrix_64ch[:, 9:10], inv_view_matrix_64ch[:, 10:11]
        ray_o_x, ray_o_y, ray_o_z = inv_view_matrix_64ch[:, 3:4], inv_view_matrix_64ch[:, 7:8], inv_view_matrix_64ch[:, 11:12]

        dx = r00 * self.cam_dx + r01 * self.cam_dy + r02 * self.cam_dz
        dy = r10 * self.cam_dx + r11 * self.cam_dy + r12 * self.cam_dz
        dz = r20 * self.cam_dx + r21 * self.cam_dy + r22 * self.cam_dz

        inv_len = torch.rsqrt(dx*dx + dy*dy + dz*dz + 1e-5)
        ray_d_x, ray_d_y, ray_d_z = dx * inv_len, dy * inv_len, dz * inv_len

        # =====================================================================
        # STAGE 2: 仮想BVH探索回路
        # =====================================================================
        boxes_reshaped = bvh_boxes_64ch[:, 0:24].view(1, 6, B, 1, 1)
        b_min_x, b_min_y, b_min_z = boxes_reshaped[:, 0], boxes_reshaped[:, 1], boxes_reshaped[:, 2]
        b_max_x, b_max_y, b_max_z = boxes_reshaped[:, 3], boxes_reshaped[:, 4], boxes_reshaped[:, 5]

        inv_rd_x, inv_rd_y, inv_rd_z = 1.0 / (ray_d_x + 1e-5), 1.0 / (ray_d_y + 1e-5), 1.0 / (ray_d_z + 1e-5)

        t1_x, t2_x = (b_min_x - ray_o_x) * inv_rd_x, (b_max_x - ray_o_x) * inv_rd_x
        t1_y, t2_y = (b_min_y - ray_o_y) * inv_rd_y, (b_max_y - ray_o_y) * inv_rd_y
        t1_z, t2_z = (b_min_z - ray_o_z) * inv_rd_z, (b_max_z - ray_o_z) * inv_rd_z

        t_near = torch.maximum(torch.maximum(torch.minimum(t1_x, t2_x), torch.minimum(t1_y, t2_y)), torch.minimum(t1_z, t2_z))
        t_far  = torch.minimum(torch.minimum(torch.maximum(t1_x, t2_x), torch.maximum(t1_y, t2_y)), torch.maximum(t1_z, t2_z))

        box_hit_all = torch.clamp(torch.relu((t_far - t_near) * 1000.0), min=0.0, max=1.0) * \
                      torch.clamp(torch.relu(t_far * 1000.0), min=0.0, max=1.0)

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
        safe_det = torch.sign(det) * torch.clamp(torch.abs(det), min=1e-4)
        inv_det = 1.0 / safe_det

        tvec_x, tvec_y, tvec_z = ray_o_x - v0_x, ray_o_y - v0_y, ray_o_z - v0_z

        u = (tvec_x * pvec_x + tvec_y * pvec_y + tvec_z * pvec_z) * inv_det
        u_valid = torch.clamp(torch.relu(u) * 1000.0, 0.0, 1.0) * torch.clamp(torch.relu(1.0 - u) * 1000.0, 0.0, 1.0)

        qvec_x = tvec_y * e1_z - tvec_z * e1_y
        qvec_y = tvec_z * e1_x - tvec_x * e1_z
        qvec_z = tvec_x * e1_y - tvec_y * e1_x

        v = (ray_d_x * qvec_x + ray_d_y * qvec_y + ray_d_z * qvec_z) * inv_det
        v_valid = torch.clamp(torch.relu(v) * 1000.0, 0.0, 1.0) * torch.clamp(torch.relu(1.0 - (u + v)) * 1000.0, 0.0, 1.0)

        poly_hit_mask = u_valid * v_valid

        safe_qvec_x = torch.clamp(qvec_x, min=-1000.0, max=1000.0)
        safe_qvec_y = torch.clamp(qvec_y, min=-1000.0, max=1000.0)
        safe_qvec_z = torch.clamp(qvec_z, min=-1000.0, max=1000.0)
        t = (e2_x * safe_qvec_x + e2_y * safe_qvec_y + e2_z * safe_qvec_z) * inv_det
        
        box_mask_for_polys = box_hit_all.repeat(1, P // B, 1, 1)
        final_poly_hit_valid = torch.clamp(torch.relu(t) * 1000.0, 0.0, 1.0) * poly_hit_mask * box_mask_for_polys

        # =====================================================================
        # STAGE 4: シェーディング & UV補間
        # =====================================================================
        n_x = e1_y * e2_z - e1_z * e2_y
        n_y = e1_z * e2_x - e1_x * e2_z
        n_z = e1_x * e2_y - e1_y * e2_x
        n_len = torch.rsqrt(n_x*n_x + n_y*n_y + n_z*n_z + 1e-5)
        n_x, n_y, n_z = n_x * n_len, n_y * n_len, n_z * n_len

        l_x, l_y, l_z = 0.5773, 0.5773, 0.5773
        diffuse = torch.clamp(n_x*l_x + n_y*l_y + n_z*l_z, min=0.0, max=1.0)

        w = 1.0 - (u + v)
        uv_u = u * 1.0 + v * 0.0 + w * 0.0
        uv_v = u * 0.0 + v * 1.0 + w * 0.0

        # =====================================================================
        # STAGE 5: Z-Buffer 出力 & 属性の統合
        # =====================================================================
        safe_t = torch.clamp(t, min=-10000.0, max=10000.0)
        t_final_buffer = (safe_t * final_poly_hit_valid) + (self.FAR_DEPTH * (1.0 - final_poly_hit_valid))

        min_depth, _ = torch.min(t_final_buffer, dim=1, keepdim=True)
        render_mask = torch.clamp(torch.relu((self.FAR_DEPTH - 1.0) - min_depth) * 1000.0, 0.0, 1.0)

        # 【ANEネイティブ対応】One-Hotマスクによる属性の抽出
        is_min = torch.clamp(torch.relu(1.0 - torch.abs(t_final_buffer - min_depth) * 1000.0), 0.0, 1.0)
        
        diffuse_out = torch.sum(diffuse * is_min, dim=1, keepdim=True) * render_mask
        uv_u_out = torch.sum(uv_u * is_min, dim=1, keepdim=True) * render_mask
        uv_v_out = torch.sum(uv_v * is_min, dim=1, keepdim=True) * render_mask

        return torch.cat([min_depth, render_mask, diffuse_out, uv_u_out, uv_v_out], dim=1)