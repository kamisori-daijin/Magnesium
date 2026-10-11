import torch
import torch.nn as nn

class ANEDynamicSDFCore(nn.Module):
    def __init__(self, width=256, height=256, max_steps=16):
        super().__init__()
        self.w, self.h = width, height
        self.max_steps = max_steps
        
        y_grid = torch.linspace(1.0, -1.0, self.h).view(1, 1, self.h, 1)
        x_grid = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        
        self.register_buffer("cam_dx", x_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dy", y_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dz", torch.full((1, 1, self.h, self.w), -1.0).half())

    def get_sdf(self, px, py, pz, type_val, size):
        # 0: Sphere
        d_sphere = torch.sqrt(px*px + py*py + pz*pz + 1e-5) - size
        
        # 1: Box
        dx, dy, dz = torch.abs(px) - size, torch.abs(py) - size, torch.abs(pz) - size
        inside = torch.clamp(torch.max(torch.max(dx, dy), dz), max=0.0)
        outside = torch.sqrt(torch.relu(dx)**2 + torch.relu(dy)**2 + torch.relu(dz)**2)
        d_box = inside + outside
        
        # 2: Capsule (Y軸に沿ったカプセル)
        py_clamped = torch.clamp(py, -size, size)
        d_capsule = torch.sqrt(px*px + (py - py_clamped)**2 + pz*pz + 1e-5) - (size * 0.5)
        
        # 3: Torus (ドーナツ型)
        q_x = torch.sqrt(px*px + pz*pz + 1e-5) - size
        d_torus = torch.sqrt(q_x*q_x + py*py + 1e-5) - (size * 0.3)

        # Ifなしで図形を選択 (One-hot的な掛け算)
        m0 = torch.clamp(1.0 - torch.abs(type_val - 0.0), min=0.0)
        m1 = torch.clamp(1.0 - torch.abs(type_val - 1.0), min=0.0)
        m2 = torch.clamp(1.0 - torch.abs(type_val - 2.0), min=0.0)
        m3 = torch.clamp(1.0 - torch.abs(type_val - 3.0), min=0.0)
        
        return m0 * d_sphere + m1 * d_box + m2 * d_capsule + m3 * d_torus

    def scene_sdf(self, px, py, pz, sdf_params):
        total_dist = torch.full_like(px, 10.0)
        
        # 最大8個の図形を処理
        for i in range(8):
            idx = i * 8
            type_val = sdf_params[:, idx:idx+1, :, :]
            op_val   = sdf_params[:, idx+1:idx+2, :, :]
            ox, oy, oz = sdf_params[:, idx+2:idx+3, :, :], sdf_params[:, idx+3:idx+4, :, :], sdf_params[:, idx+4:idx+5, :, :]
            size     = sdf_params[:, idx+5:idx+6, :, :]
            
            d = self.get_sdf(px - ox, py - oy, pz - oz, type_val, size)
            
            d_union = torch.min(total_dist, d)
            d_diff  = torch.max(total_dist, -d)
            
            total_dist = (1.0 - op_val) * d_union + op_val * d_diff
            
        return total_dist

    def forward(self, inv_view_matrix_64d, sdf_params):
        def get_mat_val(mat, idx): return mat[:, idx:idx+1, :, :]

        r00, r01, r02 = get_mat_val(inv_view_matrix_64d, 0), get_mat_val(inv_view_matrix_64d, 1), get_mat_val(inv_view_matrix_64d, 2)
        r10, r11, r12 = get_mat_val(inv_view_matrix_64d, 4), get_mat_val(inv_view_matrix_64d, 5), get_mat_val(inv_view_matrix_64d, 6)
        r20, r21, r22 = get_mat_val(inv_view_matrix_64d, 8), get_mat_val(inv_view_matrix_64d, 9), get_mat_val(inv_view_matrix_64d, 10)

        dx = r00 * self.cam_dx + r01 * self.cam_dy + r02 * self.cam_dz
        dy = r10 * self.cam_dx + r11 * self.cam_dy + r12 * self.cam_dz
        dz = r20 * self.cam_dx + r21 * self.cam_dy + r22 * self.cam_dz

        inv_len = torch.rsqrt(dx*dx + dy*dy + dz*dz + 1e-5)
        init_dx, init_dy, init_dz = dx * inv_len, dy * inv_len, dz * inv_len
        
        px, py, pz = get_mat_val(inv_view_matrix_64d, 3), get_mat_val(inv_view_matrix_64d, 7), get_mat_val(inv_view_matrix_64d, 11)

        for _ in range(self.max_steps):
            dist = self.scene_sdf(px, py, pz, sdf_params)
            px = px + init_dx * dist
            py = py + init_dy * dist
            pz = pz + init_dz * dist

        final_dist = self.scene_sdf(px, py, pz, sdf_params)
        hit_mask = torch.clamp((0.01 - final_dist) * 1000.0, min=0.0, max=1.0)
        
        return torch.cat([hit_mask] * 3, dim=1)