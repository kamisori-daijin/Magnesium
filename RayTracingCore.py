import torch
import torch.nn as nn

class ANEDynamicSDFCore(nn.Module):
    def __init__(self, width=256, height=256, max_steps=32):
        super().__init__()
        self.w, self.h = width, height
        self.max_steps = max_steps
        
        y_grid = torch.linspace(1.0, -1.0, self.h).view(1, 1, self.h, 1)
        x_grid = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        
        self.register_buffer("cam_dx", x_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dy", y_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dz", torch.full((1, 1, self.h, self.w), -1.0).half())

        self.cumsum_conv = nn.Conv2d(max_steps, max_steps, kernel_size=1, bias=False)
        weight = torch.tril(torch.ones(max_steps, max_steps)).view(max_steps, max_steps, 1, 1)
        self.cumsum_conv.weight = nn.Parameter(weight.half(), requires_grad=False)

    def get_sdf(self, px, py, pz, type_val, size):
         # 1. Sphere
         d_sphere = torch.sqrt(px*px + py*py + pz*pz + 1e-5) - size
         
         # 2. Box 
         dx = torch.abs(px) - size
         dy = torch.abs(py) - size
         dz = torch.abs(pz) - size
         outside = torch.sqrt(torch.relu(dx)**2 + torch.relu(dy)**2 + torch.relu(dz)**2)
         
    
         d_max = dx + torch.relu(dy - dx)
         d_max = d_max + torch.relu(dz - d_max)
         inside = d_max - torch.relu(d_max)
         
         d_box = outside + inside
         
         # 3. Capsule
         half_len = size * 0.5
         py_clamped = torch.clamp(py, -half_len, half_len)
         d_capsule = torch.sqrt(px*px + (py - py_clamped)**2 + pz*pz + 1e-5) - (size * 0.3)
         
         # 4. Torus
         q_x = torch.sqrt(px*px + pz*pz + 1e-5) - size
         d_torus = torch.sqrt(q_x*q_x + py*py + 1e-5) - (size * 0.3)
 
         
         m0 = torch.relu(1.0 - torch.abs(type_val - 0.0))
         m1 = torch.relu(1.0 - torch.abs(type_val - 1.0))
         m2 = torch.relu(1.0 - torch.abs(type_val - 2.0))
         m3 = torch.relu(1.0 - torch.abs(type_val - 3.0))
         
         return m0 * d_sphere + m1 * d_box + m2 * d_capsule + m3 * d_torus

    def scene_sdf(self, px, py, pz, sdf_params):
        B, _, H, W = sdf_params.shape
        params = sdf_params.view(B, 8, 8, H, W)
        
        type_val = params[:, :, 0:1, :, :]
        ox       = params[:, :, 2:3, :, :]
        oy       = params[:, :, 3:4, :, :]
        oz       = params[:, :, 4:5, :, :]
        size     = params[:, :, 5:6, :, :]

        d = self.get_sdf(
            px.unsqueeze(2) - ox.transpose(1, 2), 
            py.unsqueeze(2) - oy.transpose(1, 2), 
            pz.unsqueeze(2) - oz.transpose(1, 2), 
            type_val.transpose(1, 2), 
            size.transpose(1, 2)
        )

        final_d, _ = torch.min(d, dim=2, keepdim=False)
        return final_d

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

        step_offsets = torch.linspace(0.0, 2.5, self.max_steps, device=px.device).view(1, self.max_steps, 1, 1).half()
        
        px_c = px.expand(1, self.max_steps, self.h, self.w) + init_dx * step_offsets
        py_c = py.expand(1, self.max_steps, self.h, self.w) + init_dy * step_offsets
        pz_c = pz.expand(1, self.max_steps, self.h, self.w) + init_dz * step_offsets

        dists = self.scene_sdf(px_c, py_c, pz_c, sdf_params) * 0.3
        accumulated_dists = self.cumsum_conv(dists)

        px_final = px_c + init_dx * accumulated_dists
        py_final = py_c + init_dy * accumulated_dists
        pz_final = pz_c + init_dz * accumulated_dists

        final_dist = self.scene_sdf(px_final, py_final, pz_final, sdf_params)
        
        min_dist, _ = torch.min(final_dist, dim=1, keepdim=True)
        
        hit_mask = torch.relu((0.01 - min_dist) * 1000.0)
        hit_mask = 1.0 - torch.relu(1.0 - hit_mask)
        
        return hit_mask.expand(1, 3, -1, -1)