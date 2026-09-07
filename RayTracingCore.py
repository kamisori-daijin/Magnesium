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
        self.eps = 0.02  # Minimum step size for numerical differentiation
        
        # Ray
        y_grid = torch.linspace(1.0, -1.0, self.h).view(1, 1, self.h, 1)
        x_grid = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        
        self.register_buffer("cam_dx", x_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dy", y_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dz", torch.full((1, 1, self.h, self.w), -1.0).half())
        
        self.floor_y = -0.8
        self.register_buffer("ONES", torch.ones(1, 1, self.h, self.w).half())
        self.register_buffer("ZEROS", torch.zeros(1, 1, self.h, self.w).half())
        
        # World space light direction
        light_dir_x = torch.full((1, 1, 1, 1), 1.0)
        light_dir_y = torch.full((1, 1, 1, 1), 1.0)
        light_dir_z = torch.full((1, 1, 1, 1), 1.0)
        inv_l_len = torch.rsqrt(light_dir_x*light_dir_x + light_dir_y*light_dir_y + light_dir_z*light_dir_z + 1e-5)
        self.register_buffer("light_dx", (light_dir_x * inv_l_len).half())
        self.register_buffer("light_dy", (light_dir_y * inv_l_len).half())
        self.register_buffer("light_dz", (light_dir_z * inv_l_len).half())

        # Dimension
        self.register_buffer("step_ratios", (torch.arange(self.max_steps).view(1, self.max_steps, 1, 1) * self.dt).half())
        self.register_buffer("shadow_ratios", (torch.arange(self.shadow_steps).view(1, self.shadow_steps, 1, 1) * self.dt).half())

        # Similer RT core hack: 
        y_tex = torch.linspace(-1.0, 1.0, self.h).view(1, 1, self.h, 1)
        x_tex = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        base_mask_x = torch.clamp(1.0 - (torch.abs(x_tex) - 0.4) * 100.0, min=0.0, max=1.0)
        base_mask_y = torch.clamp(1.0 - (torch.abs(y_tex) - 0.4) * 100.0, min=0.0, max=1.0)
        cube_2d_mask = (base_mask_x * base_mask_y).half()

        # XY, XZ, YZ
        self.register_buffer("base_multiview_textures", torch.cat([cube_2d_mask, cube_2d_mask, cube_2d_mask], dim=1))

    def check_multiview_hit(self, px, py, pz):
        """
        px, py, pz: warp coordinates
        """
        # Box boundary check
        out_x = torch.relu(torch.abs(px) - 1.0)
        out_y = torch.relu(torch.abs(py) - 1.0)
        out_z = torch.relu(torch.abs(pz) - 1.0)
        any_out = torch.clamp((out_x + out_y + out_z) * 100.0, min=0.0, max=1.0)
        box_check = 1.0 - any_out

        # Smooth cube function (high-order power hack) to avoid jagged silhouette edges
        proj_xy = torch.clamp(1.1 - (px.pow(4) + py.pow(4)) / 0.35, min=0.0, max=1.0)
        proj_xz = torch.clamp(1.1 - (px.pow(4) + pz.pow(4)) / 0.35, min=0.0, max=1.0)
        proj_yz = torch.clamp(1.1 - (py.pow(4) + pz.pow(4)) / 0.35, min=0.0, max=1.0)

        # Masking
        mask_xy = self.base_multiview_textures[:, 0:1, :, :] * proj_xy
        mask_xz = self.base_multiview_textures[:, 1:2, :, :] * proj_xz
        mask_yz = self.base_multiview_textures[:, 2:3, :, :] * proj_yz

        object_hit = mask_xy * mask_xz * mask_yz * box_check
        return object_hit

    def forward(self, multiview_textures, inv_view_matrix_64d):
        """
        Input:
          multiview_textures: [1, 3, 256, 256] 
          inv_view_matrix_64d: [1, 64, 1, 1] 
        """
       
        # 16ch Restore camera inverse matrix
        inv_view = inv_view_matrix_64d[0, :16, 0, 0].view(4, 4).half()

        # 16ch Restore object inverse model matrix
        inv_model = inv_view_matrix_64d[0, 16:32, 0, 0].view(4, 4).half()

        # 16ch Restore camera matrix
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
        #  2. Ray tracing
        # ==========================================
        # [1, max_steps, H, W] Broadcast
        px_all = init_px + init_dx * self.step_ratios
        py_all = init_py + init_dy * self.step_ratios
        pz_all = init_pz + init_dz * self.step_ratios

        # 3. Ray marching
        m00, m01, m02, m03 = inv_model[0, 0], inv_model[0, 1], inv_model[0, 2], inv_model[0, 3]
        m10, m11, m12, m13 = inv_model[1, 0], inv_model[1, 1], inv_model[1, 2], inv_model[1, 3]
        m20, m21, m22, m23 = inv_model[2, 0], inv_model[2, 1], inv_model[2, 2], inv_model[2, 3]

        local_px_all = m00 * px_all + m01 * py_all + m02 * pz_all + m03
        local_py_all = m10 * px_all + m11 * py_all + m12 * pz_all + m13
        local_pz_all = m20 * px_all + m21 * py_all + m22 * pz_all + m23

        
        object_hit_all = self.check_multiview_hit(local_px_all, local_py_all, local_pz_all)

        # py: floor height
        floor_hit_all = torch.clamp(torch.relu(self.floor_y - py_all) * 100.0, min=0.0, max=1.0)
        any_hit_all = torch.clamp(object_hit_all + floor_hit_all, min=0.0, max=1.0)

        # Cumulative hit
        cum_hit = torch.cumsum(any_hit_all, dim=1)

        prior_hit = torch.cat([torch.zeros_like(cum_hit[:, :1, :, :]), cum_hit[:, :-1, :, :]], dim=1)
        not_hit_yet_all = torch.clamp(1.0 - prior_hit, min=0.0, max=1.0)

        is_first_object_all = not_hit_yet_all * object_hit_all
        is_first_floor_all = not_hit_yet_all * (1.0 - object_hit_all) * floor_hit_all

        # Dim 1（Channel）[1, 1, H, W]
        hit_object_mask = torch.sum(is_first_object_all, dim=1, keepdim=True).clamp(0.0, 1.0)
        hit_floor_mask = torch.sum(is_first_floor_all, dim=1, keepdim=True).clamp(0.0, 1.0)
        accum_hit = torch.clamp(hit_object_mask + hit_floor_mask, min=0.0, max=1.0)

        # World space coordinates
        px = torch.sum(is_first_object_all * px_all + is_first_floor_all * px_all, dim=1, keepdim=True)
        py = torch.sum(is_first_object_all * py_all + is_first_floor_all * py_all, dim=1, keepdim=True)
        pz = torch.sum(is_first_object_all * pz_all + is_first_floor_all * pz_all, dim=1, keepdim=True)

        # ==========================================
        # 3. calculate normal vector
        # ==========================================
        # Local space coordinates
        local_px = m00 * px + m01 * py + m02 * pz + m03
        local_py = m10 * px + m11 * py + m12 * pz + m13
        local_pz = m20 * px + m21 * py + m22 * pz + m23

        # Calculate normal vector using central difference
        f_center = self.check_multiview_hit(local_px, local_py, local_pz)
        f_dx = self.check_multiview_hit(local_px + self.eps, local_py, local_pz)
        f_dy = self.check_multiview_hit(local_px, local_py + self.eps, local_pz)
        f_dz = self.check_multiview_hit(local_px, local_py, local_pz + self.eps)

        raw_nx = f_center - f_dx
        raw_ny = f_center - f_dy
        raw_nz = f_center - f_dz

      
        # Restore model matrix
        world_nx = m00 * raw_nx + m10 * raw_ny + m20 * raw_nz
        world_ny = m01 * raw_nx + m11 * raw_ny + m21 * raw_nz
        world_nz = m02 * raw_nx + m12 * raw_ny + m22 * raw_nz

        inv_true_n_len = torch.rsqrt(world_nx*world_nx + world_ny*world_ny + world_nz*world_nz + 1e-5)

        # World space normal vector
        first_nx = hit_object_mask * (world_nx * inv_true_n_len)
        first_ny = hit_object_mask * (world_ny * inv_true_n_len) + hit_floor_mask * 1.0
        first_nz = hit_object_mask * (world_nz * inv_true_n_len)

        # ==========================================
        # Shadow ray
        # ==========================================
        shadow_start_x = px + 0.04 * self.light_dx
        shadow_start_y = py + 0.04 * self.light_dy
        shadow_start_z = pz + 0.04 * self.light_dz

        # [1, shadow_steps, H, W] Broadcast
        spx_all = shadow_start_x + self.light_dx * self.shadow_ratios
        spy_all = shadow_start_y + self.light_dy * self.shadow_ratios
        spz_all = shadow_start_z + self.light_dz * self.shadow_ratios

        # warp to local space
        local_spx_all = m00 * spx_all + m01 * spy_all + m02 * spz_all + m03
        local_spy_all = m10 * spx_all + m11 * spy_all + m12 * spz_all + m13
        local_spz_all = m20 * spx_all + m21 * spy_all + m22 * spz_all + m23
        shadow_hit_all = self.check_multiview_hit(local_spx_all, local_spy_all, local_spz_all)
        accum_shadow = torch.sum(shadow_hit_all, dim=1, keepdim=True).clamp(0.0, 1.0) * hit_floor_mask

        # ==========================================
        # Checker and lighting
        # ==========================================
        diffuse = first_nx * self.light_dx + first_ny * self.light_dy + first_nz * self.light_dz
        shading = torch.relu(diffuse) + 0.15
        
        sign_x = torch.clamp(px * 3.0 * 100.0, min=-1.0, max=1.0)
        sign_z = torch.clamp(pz * 3.0 * 100.0, min=-1.0, max=1.0)
        checker = (sign_x * sign_z + 1.0) * 0.5
        floor_color = hit_floor_mask * (0.3 + 0.2 * checker)
        base_color = hit_object_mask * self.ONES + floor_color
        light_modifier = (self.ONES - accum_shadow) * shading + accum_shadow * 0.15

        return accum_hit * base_color * light_modifier
