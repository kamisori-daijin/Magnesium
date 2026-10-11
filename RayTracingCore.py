import torch
import torch.nn as nn

class ANEDynamicRayTracingCore(nn.Module):
    def __init__(self, width=256, height=256, max_steps=64, grid_size=8):
        super().__init__()
        self.w, self.h = width, height
        self.max_steps = max_steps
        self.dt = 0.05
        self.grid_size = grid_size
        self.num_voxels = grid_size ** 3
        
        self.register_buffer("eps", torch.tensor([[[[0.05]]]]).half())
        self.register_buffer("floor_y", torch.tensor([[[[-0.8]]]]).half())
        
        y_grid = torch.linspace(1.0, -1.0, self.h).view(1, 1, self.h, 1)
        x_grid = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        
        self.register_buffer("cam_dx", x_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dy", y_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dz", torch.full((1, 1, self.h, self.w), -1.0).half())
        
        self.register_buffer("light_dx", torch.tensor([[[[0.5773]]]]).half())
        self.register_buffer("light_dy", torch.tensor([[[[0.5773]]]]).half())
        self.register_buffer("light_dz", torch.tensor([[[[0.5773]]]]).half())
        
        self.register_buffer("step_ratios", (torch.arange(self.max_steps).view(1, self.max_steps, 1, 1) * self.dt).half())
        self.register_buffer("shadow_ratios", (torch.arange(16).view(1, 16, 1, 1) * self.dt).half())

        # 座標グリッド
        coords = torch.linspace(-1.0, 1.0, self.grid_size).half()
        grid_z, grid_y, grid_x = torch.meshgrid(coords, coords, coords, indexing='ij')
        self.register_buffer("grid_x", grid_x.reshape(1, -1, 1, 1))
        self.register_buffer("grid_y", grid_y.reshape(1, -1, 1, 1))
        self.register_buffer("grid_z", grid_z.reshape(1, -1, 1, 1))

        # ANE用 累積和 Conv2d
        self.ane_cumsum_conv = nn.Conv2d(self.max_steps, self.max_steps, kernel_size=1, bias=False)
        self.ane_cumsum_conv.weight.data = torch.tril(torch.ones(self.max_steps, self.max_steps)).view(self.max_steps, self.max_steps, 1, 1).half()
        self.ane_cumsum_conv.weight.requires_grad = False

        # 動的ルックアップ用 1x1 Conv2d
        self.lookup_conv = nn.Conv2d(self.num_voxels, 1, kernel_size=1, bias=False)
        self.lookup_conv.weight.requires_grad = False

    def check_voxel_hit(self, voxel_data, px, py, pz):
        eps = torch.tensor([[[[2.0 / self.grid_size]]]], device=px.device, dtype=px.dtype)
        inv_eps = torch.reciprocal(eps)
        
        mask_x = torch.relu(1.0 - torch.abs(px - self.grid_x) * inv_eps)
        mask_y = torch.relu(1.0 - torch.abs(py - self.grid_y) * inv_eps)
        mask_z = torch.relu(1.0 - torch.abs(pz - self.grid_z) * inv_eps)
        
        indices = mask_x * mask_y * mask_z
        
        self.lookup_conv.weight.data = voxel_data.view(1, self.num_voxels, 1, 1)
        return self.lookup_conv(indices)

    def forward(self, voxel_data, inv_view_matrix_64d):
        def get_mat_val(mat, idx): return mat[:, idx:idx+1, :, :]

        r00, r01, r02 = get_mat_val(inv_view_matrix_64d, 0), get_mat_val(inv_view_matrix_64d, 1), get_mat_val(inv_view_matrix_64d, 2)
        r10, r11, r12 = get_mat_val(inv_view_matrix_64d, 4), get_mat_val(inv_view_matrix_64d, 5), get_mat_val(inv_view_matrix_64d, 6)
        r20, r21, r22 = get_mat_val(inv_view_matrix_64d, 8), get_mat_val(inv_view_matrix_64d, 9), get_mat_val(inv_view_matrix_64d, 10)

        dx = r00 * self.cam_dx + r01 * self.cam_dy + r02 * self.cam_dz
        dy = r10 * self.cam_dx + r11 * self.cam_dy + r12 * self.cam_dz
        dz = r20 * self.cam_dx + r21 * self.cam_dy + r22 * self.cam_dz

        inv_len = torch.rsqrt(dx*dx + dy*dy + dz*dz + 1e-5)
        init_dx, init_dy, init_dz = dx * inv_len, dy * inv_len, dz * inv_len
        init_px, init_py, init_pz = get_mat_val(inv_view_matrix_64d, 3), get_mat_val(inv_view_matrix_64d, 7), get_mat_val(inv_view_matrix_64d, 11)

        px_all, py_all, pz_all = init_px + init_dx * self.step_ratios, init_py + init_dy * self.step_ratios, init_pz + init_dz * self.step_ratios

        # 各ステップのボクセル判定
        object_hit_all = torch.cat([self.check_voxel_hit(voxel_data, px_all[:, i:i+1], py_all[:, i:i+1], pz_all[:, i:i+1]) for i in range(self.max_steps)], dim=1)
        
        floor_hit_all = torch.clamp(torch.relu(self.floor_y - py_all) * 100.0, min=0.0, max=1.0)
        any_hit_all = torch.clamp(object_hit_all + floor_hit_all, min=0.0, max=1.0)

        cum_hit = self.ane_cumsum_conv(any_hit_all)
        not_hit_yet_all = torch.clamp(1.0 - torch.cat([torch.zeros_like(cum_hit[:, :1, :, :]), cum_hit[:, :-1, :, :]], dim=1), min=0.0, max=1.0)

        is_first_object_all = not_hit_yet_all * object_hit_all
        is_first_floor_all = not_hit_yet_all * (1.0 - object_hit_all) * floor_hit_all

        hit_object_mask = torch.sum(is_first_object_all, dim=1, keepdim=True).clamp(0.0, 1.0)
        hit_floor_mask = torch.sum(is_first_floor_all, dim=1, keepdim=True).clamp(0.0, 1.0)

        px = torch.sum(is_first_object_all * px_all + is_first_floor_all * px_all, dim=1, keepdim=True)
        py = torch.sum(is_first_object_all * py_all + is_first_floor_all * py_all, dim=1, keepdim=True)
        pz = torch.sum(is_first_object_all * pz_all + is_first_floor_all * pz_all, dim=1, keepdim=True)

        f_c = self.check_voxel_hit(voxel_data, px, py, pz)
        f_x = self.check_voxel_hit(voxel_data, px + self.eps, py, pz)
        f_y = self.check_voxel_hit(voxel_data, px, py + self.eps, pz)
        f_z = self.check_voxel_hit(voxel_data, px, py, pz + self.eps)

        nx, ny, nz = f_c - f_x, f_c - f_y, f_c - f_z
        inv_n_len = torch.rsqrt(nx*nx + ny*ny + nz*nz + 1e-5)
        
        first_nx = hit_object_mask * (nx * inv_n_len)
        first_ny = hit_object_mask * (ny * inv_n_len) + hit_floor_mask * 1.0
        first_nz = hit_object_mask * (nz * inv_n_len)

        spx_all = px + 0.04 * self.light_dx + self.light_dx * self.shadow_ratios
        spy_all = py + 0.04 * self.light_dy + self.light_dy * self.shadow_ratios
        spz_all = pz + 0.04 * self.light_dz + self.light_dz * self.shadow_ratios
        
        s_hit = torch.cat([self.check_voxel_hit(voxel_data, spx_all[:, i:i+1], spy_all[:, i:i+1], spz_all[:, i:i+1]) for i in range(16)], dim=1)
        accum_shadow = torch.sum(s_hit, dim=1, keepdim=True).clamp(0.0, 1.0) * hit_floor_mask

        diffuse = torch.relu(first_nx * self.light_dx + first_ny * self.light_dy + first_nz * self.light_dz)
        shading = diffuse + 0.15
        light_modifier = (1.0 - accum_shadow) * shading + accum_shadow * 0.15

        base_color = hit_object_mask * 0.8 + hit_floor_mask * 0.3
        final_color = base_color * light_modifier

        return torch.cat([final_color] * 3, dim=1)