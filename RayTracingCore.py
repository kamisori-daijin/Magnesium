import torch
import torch.nn as nn

class ANERayTracingCore(nn.Module):
    def __init__(self, width=256, height=256, max_steps=64, shadow_steps=16):
        super().__init__()
        self.w = width
        self.h = height
        self.max_steps = max_steps
        self.shadow_steps = shadow_steps
        self.dt = 0.08
        
        self.register_buffer("eps", torch.tensor([[[[0.02]]]]).half())
        self.register_buffer("floor_y", torch.tensor([[[[-0.8]]]]).half())
        
        y_grid = torch.linspace(1.0, -1.0, self.h).view(1, 1, self.h, 1)
        x_grid = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        
        self.register_buffer("cam_dx", x_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dy", y_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dz", torch.full((1, 1, self.h, self.w), -1.0).half())
        
        self.register_buffer("ONES", torch.ones(1, 1, self.h, self.w).half())
        self.register_buffer("ZEROS", torch.zeros(1, 1, self.h, self.w).half())
        
        self.register_buffer("light_dx", torch.tensor([[[[0.5773]]]]).half())
        self.register_buffer("light_dy", torch.tensor([[[[0.5773]]]]).half())
        self.register_buffer("light_dz", torch.tensor([[[[0.5773]]]]).half())

        self.register_buffer("step_ratios", (torch.arange(self.max_steps).view(1, self.max_steps, 1, 1) * self.dt).half())
        self.register_buffer("shadow_ratios", (torch.arange(self.shadow_steps).view(1, self.shadow_steps, 1, 1) * self.dt).half())

        y_tex = torch.linspace(-1.0, 1.0, self.h).view(1, 1, self.h, 1)
        x_tex = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        base_mask_x = torch.clamp(1.0 - (torch.abs(x_tex) - 0.4) * 100.0, min=0.0, max=1.0)
        base_mask_y = torch.clamp(1.0 - (torch.abs(y_tex) - 0.4) * 100.0, min=0.0, max=1.0)
        cube_2d_mask = (base_mask_x * base_mask_y).half()

        self.register_buffer("base_multiview_textures", torch.cat([cube_2d_mask, cube_2d_mask, cube_2d_mask], dim=1))

        # 🌟 1. 累積和用 Conv2d
        self.ane_cumsum_conv = nn.Conv2d(self.max_steps, self.max_steps, kernel_size=1, bias=False)
        weight_matrix = torch.tril(torch.ones(self.max_steps, self.max_steps))
        self.ane_cumsum_conv.weight.data = weight_matrix.view(self.max_steps, self.max_steps, 1, 1).half()
        self.ane_cumsum_conv.weight.requires_grad = False

        # 🌟 2. マスク結合用 Conv2d (入力3 -> 出力1)
        self.mask_combine_conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, bias=False)
        nn.init.constant_(self.mask_combine_conv.weight, 1.0)
        self.mask_combine_conv.weight.data = self.mask_combine_conv.weight.data.half()
        self.mask_combine_conv.weight.requires_grad = False

    def check_multiview_hit(self, px, py, pz):
        out_x = torch.relu(torch.abs(px) - 1.0)
        out_y = torch.relu(torch.abs(py) - 1.0)
        out_z = torch.relu(torch.abs(pz) - 1.0)
        any_out = torch.clamp((out_x + out_y + out_z) * 100.0, min=0.0, max=1.0)
        box_check = 1.0 - any_out

        proj_xy = torch.clamp(1.0 - torch.relu(torch.abs(px) - 0.4) * 10.0, 0.0, 1.0) * \
                  torch.clamp(1.0 - torch.relu(torch.abs(py) - 0.4) * 10.0, 0.0, 1.0)
        proj_xz = torch.clamp(1.0 - torch.relu(torch.abs(px) - 0.4) * 10.0, 0.0, 1.0) * \
                  torch.clamp(1.0 - torch.relu(torch.abs(pz) - 0.4) * 10.0, 0.0, 1.0)
        proj_yz = torch.clamp(1.0 - torch.relu(torch.abs(py) - 0.4) * 10.0, 0.0, 1.0) * \
                  torch.clamp(1.0 - torch.relu(torch.abs(pz) - 0.4) * 10.0, 0.0, 1.0)

        mask_xy = self.base_multiview_textures[:, 0:1, :, :] * proj_xy
        mask_xz = self.base_multiview_textures[:, 1:2, :, :] * proj_xz
        mask_yz = self.base_multiview_textures[:, 2:3, :, :] * proj_yz

        steps = px.size(1)

        # 🌟 ステップ数をバッチ次元に逃がして結合
        combined_masks = torch.cat([
            mask_xy.view(steps, 1, self.h, self.w),
            mask_xz.view(steps, 1, self.h, self.w),
            mask_yz.view(steps, 1, self.h, self.w)
        ], dim=1)

        object_hit_raw = self.mask_combine_conv(combined_masks)
        object_hit_raw = object_hit_raw.view(1, steps, self.h, self.w)
        
        object_hit = torch.clamp(object_hit_raw - 2.0, min=0.0, max=1.0) * box_check
        return object_hit

    def forward(self, multiview_textures, inv_view_matrix_64d):
        def get_mat_val(mat, idx):
            return mat[:, idx:idx+1, :, :]

        r00, r01, r02 = get_mat_val(inv_view_matrix_64d, 0), get_mat_val(inv_view_matrix_64d, 1), get_mat_val(inv_view_matrix_64d, 2)
        r10, r11, r12 = get_mat_val(inv_view_matrix_64d, 4), get_mat_val(inv_view_matrix_64d, 5), get_mat_val(inv_view_matrix_64d, 6)
        r20, r21, r22 = get_mat_val(inv_view_matrix_64d, 8), get_mat_val(inv_view_matrix_64d, 9), get_mat_val(inv_view_matrix_64d, 10)

        dx = r00 * self.cam_dx + r01 * self.cam_dy + r02 * self.cam_dz
        dy = r10 * self.cam_dx + r11 * self.cam_dy + r12 * self.cam_dz
        dz = r20 * self.cam_dx + r21 * self.cam_dy + r22 * self.cam_dz

        inv_len = torch.rsqrt(dx*dx + dy*dy + dz*dz + 1e-5)
        init_dx, init_dy, init_dz = dx * inv_len, dy * inv_len, dz * inv_len

        init_px, init_py, init_pz = get_mat_val(inv_view_matrix_64d, 3), get_mat_val(inv_view_matrix_64d, 7), get_mat_val(inv_view_matrix_64d, 11)

        px_all = init_px + init_dx * self.step_ratios
        py_all = init_py + init_dy * self.step_ratios
        pz_all = init_pz + init_dz * self.step_ratios

        m1_00, m1_01, m1_02, m1_03 = get_mat_val(inv_view_matrix_64d, 16), get_mat_val(inv_view_matrix_64d, 17), get_mat_val(inv_view_matrix_64d, 18), get_mat_val(inv_view_matrix_64d, 19)
        m1_10, m1_11, m1_12, m1_13 = get_mat_val(inv_view_matrix_64d, 20), get_mat_val(inv_view_matrix_64d, 21), get_mat_val(inv_view_matrix_64d, 22), get_mat_val(inv_view_matrix_64d, 23)
        m1_20, m1_21, m1_22, m1_23 = get_mat_val(inv_view_matrix_64d, 24), get_mat_val(inv_view_matrix_64d, 25), get_mat_val(inv_view_matrix_64d, 26), get_mat_val(inv_view_matrix_64d, 27)

        local1_px_all = m1_00 * px_all + m1_01 * py_all + m1_02 * pz_all + m1_03
        local1_py_all = m1_10 * px_all + m1_11 * py_all + m1_12 * pz_all + m1_13
        local1_pz_all = m1_20 * px_all + m1_21 * py_all + m1_22 * pz_all + m1_23

        m2_00, m2_01, m2_02, m2_03 = get_mat_val(inv_view_matrix_64d, 32), get_mat_val(inv_view_matrix_64d, 33), get_mat_val(inv_view_matrix_64d, 34), get_mat_val(inv_view_matrix_64d, 35)
        m2_10, m2_11, m2_12, m2_13 = get_mat_val(inv_view_matrix_64d, 36), get_mat_val(inv_view_matrix_64d, 37), get_mat_val(inv_view_matrix_64d, 38), get_mat_val(inv_view_matrix_64d, 39)
        m2_20, m2_21, m2_22, m2_23 = get_mat_val(inv_view_matrix_64d, 40), get_mat_val(inv_view_matrix_64d, 41), get_mat_val(inv_view_matrix_64d, 42), get_mat_val(inv_view_matrix_64d, 43)

        local2_px_all = m2_00 * px_all + m2_01 * py_all + m2_02 * pz_all + m2_03
        local2_py_all = m2_10 * px_all + m2_11 * py_all + m2_12 * pz_all + m2_13
        local2_pz_all = m2_20 * px_all + m2_21 * py_all + m2_22 * pz_all + m2_23

        hit1_all = self.check_multiview_hit(local1_px_all, local1_py_all, local1_pz_all)
        hit2_all = self.check_multiview_hit(local2_px_all, local2_py_all, local2_pz_all)
        object_hit_all = torch.max(hit1_all, hit2_all)

        floor_hit_all = torch.clamp(torch.relu(self.floor_y - py_all) * 100.0, min=0.0, max=1.0)
        any_hit_all = torch.clamp(object_hit_all + floor_hit_all, min=0.0, max=1.0)

        cum_hit = self.ane_cumsum_conv(any_hit_all)

        prior_hit = torch.cat([torch.zeros_like(cum_hit[:, :1, :, :]), cum_hit[:, :-1, :, :]], dim=1)
        not_hit_yet_all = torch.clamp(1.0 - prior_hit, min=0.0, max=1.0)

        is_first_object_all = not_hit_yet_all * object_hit_all
        is_first_floor_all = not_hit_yet_all * (1.0 - object_hit_all) * floor_hit_all

        hit_object_mask = torch.sum(is_first_object_all, dim=1, keepdim=True).clamp(0.0, 1.0)
        hit_floor_mask = torch.sum(is_first_floor_all, dim=1, keepdim=True).clamp(0.0, 1.0)
        accum_hit = torch.clamp(hit_object_mask + hit_floor_mask, min=0.0, max=1.0)

        px = torch.sum(is_first_object_all * px_all + is_first_floor_all * px_all, dim=1, keepdim=True)
        py = torch.sum(is_first_object_all * py_all + is_first_floor_all * py_all, dim=1, keepdim=True)
        pz = torch.sum(is_first_object_all * pz_all + is_first_floor_all * pz_all, dim=1, keepdim=True)

        local1_px = m1_00 * px + m1_01 * py + m1_02 * pz + m1_03
        local1_py = m1_10 * px + m1_11 * py + m1_12 * pz + m1_13
        local1_pz = m1_20 * px + m1_21 * py + m1_22 * pz + m1_23

        local2_px = m2_00 * px + m2_01 * py + m2_02 * pz + m2_03
        local2_py = m2_10 * px + m2_11 * py + m2_12 * pz + m2_13
        local2_pz = m2_20 * px + m2_21 * py + m2_22 * pz + m2_23

        f1_c, f2_c = self.check_multiview_hit(local1_px, local1_py, local1_pz), self.check_multiview_hit(local2_px, local2_py, local2_pz)
        f1_x, f2_x = self.check_multiview_hit(local1_px + self.eps, local1_py, local1_pz), self.check_multiview_hit(local2_px + self.eps, local2_py, local2_pz)
        f1_y, f2_y = self.check_multiview_hit(local1_px, local1_py + self.eps, local1_pz), self.check_multiview_hit(local2_px, local2_py + self.eps, local2_pz)
        f1_z, f2_z = self.check_multiview_hit(local1_px, local1_py, local1_pz + self.eps), self.check_multiview_hit(local2_px, local2_py, local2_pz + self.eps)

        obj1_mask = torch.sum(is_first_object_all * hit1_all, dim=1, keepdim=True).clamp(0.0, 1.0)
        obj2_mask = torch.sum(is_first_object_all * (1.0 - hit1_all) * hit2_all, dim=1, keepdim=True).clamp(0.0, 1.0)

        raw_nx1, raw_ny1, raw_nz1 = f1_c - f1_x, f1_c - f1_y, f1_c - f1_z
        raw_nx2, raw_ny2, raw_nz2 = f2_c - f2_x, f2_c - f2_y, f2_c - f2_z

        world_nx = obj1_mask * (m1_00 * raw_nx1 + m1_10 * raw_ny1 + m1_20 * raw_nz1) + obj2_mask * (m2_00 * raw_nx2 + m2_10 * raw_ny2 + m2_20 * raw_nz2)
        world_ny = obj1_mask * (m1_01 * raw_nx1 + m1_11 * raw_ny1 + m1_21 * raw_nz1) + obj2_mask * (m2_01 * raw_nx2 + m2_11 * raw_ny2 + m2_21 * raw_nz2)
        world_nz = obj1_mask * (m1_02 * raw_nx1 + m1_12 * raw_ny1 + m1_22 * raw_nz1) + obj2_mask * (m2_02 * raw_nx2 + m2_12 * raw_ny2 + m2_22 * raw_nz2)

        inv_true_n_len = torch.rsqrt(world_nx*world_nx + world_ny*world_ny + world_nz*world_nz + 1e-5)
        
        first_nx = hit_object_mask * (world_nx * inv_true_n_len)
        first_ny = hit_object_mask * (world_ny * inv_true_n_len) + hit_floor_mask * 1.0
        first_nz = hit_object_mask * (world_nz * inv_true_n_len)

        shadow_start_x, shadow_start_y, shadow_start_z = px + 0.04 * self.light_dx, py + 0.04 * self.light_dy, pz + 0.04 * self.light_dz
        spx_all = shadow_start_x + self.light_dx * self.shadow_ratios
        spy_all = shadow_start_y + self.light_dy * self.shadow_ratios
        spz_all = shadow_start_z + self.light_dz * self.shadow_ratios

        l1_spx = m1_00 * spx_all + m1_01 * spy_all + m1_02 * spz_all + m1_03
        l1_spy = m1_10 * spx_all + m1_11 * spy_all + m1_12 * spz_all + m1_13
        l1_spz = m1_20 * spx_all + m1_21 * spy_all + m1_22 * spz_all + m1_23

        l2_spx = m2_00 * spx_all + m2_01 * spy_all + m2_02 * spz_all + m2_03
        l2_spy = m2_10 * spx_all + m2_11 * spy_all + m2_12 * spz_all + m2_13
        l2_spz = m2_20 * spx_all + m2_21 * spy_all + m2_22 * spz_all + m2_23

        s_hit1 = self.check_multiview_hit(l1_spx, l1_spy, l1_spz)
        s_hit2 = self.check_multiview_hit(l2_spx, l2_spy, l2_spz)
        accum_shadow = torch.sum(torch.max(s_hit1, s_hit2), dim=1, keepdim=True).clamp(0.0, 1.0) * hit_floor_mask

        diffuse = first_nx * self.light_dx + first_ny * self.light_dy + first_nz * self.light_dz
        shading = torch.relu(diffuse) + 0.15

        sign_x = torch.clamp(px * 3.0 * 100.0, min=-1.0, max=1.0)
        sign_z = torch.clamp(pz * 3.0 * 100.0, min=-1.0, max=1.0)
        checker = (sign_x * sign_z + 1.0) * 0.5
        floor_color = hit_floor_mask * (0.3 + 0.2 * checker)

        obj1_color = obj1_mask * self.ONES
        rgb_object_color = torch.cat([obj1_color + obj2_mask * 0.7, obj1_color + obj2_mask * 0.8, obj1_color + obj2_mask * 1.0], dim=1)
        rgb_floor_color = torch.cat([floor_color, floor_color, floor_color], dim=1)

        base_color = rgb_object_color + rgb_floor_color
        light_modifier = (self.ONES - accum_shadow) * shading + accum_shadow * 0.15

        return accum_hit * base_color * light_modifier