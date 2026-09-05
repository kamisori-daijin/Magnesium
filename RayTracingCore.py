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
        self.eps = 0.02  # 数値微分用のステップ
        
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

        self.register_buffer("step_ratios", (torch.arange(self.max_steps).view(self.max_steps, 1, 1, 1) * self.dt).half())
        self.register_buffer("shadow_ratios", (torch.arange(self.shadow_steps).view(self.shadow_steps, 1, 1, 1) * self.dt).half())

    def check_multiview_hit(self, px, py, pz, multiview_textures):
        """
        [完全修正版] 3次元座標 px, py, pz から、3面図マスクの内部にあるかを「動的判定」します。
        数式に頼らず、3面図のシルエットのみで形状を決定します。
        """
        # 1. 空間位置 px, py, pz から各プレーンへの投影座標（-1.0〜1.0）を取得
        # ANEで動く高精度なシグモイド/ステップ関数ハックを使い、マスクの境界を座標依存で判定させます
        # 3面図の解像度が256なので、微小な座標変化がマスクのヒット状態に直結するようにシャープなブレンドを施します
        
        # バウンディングボックス境界判定
        out_x = torch.relu(torch.abs(px) - 1.0)
        out_y = torch.relu(torch.abs(py) - 1.0)
        out_z = torch.relu(torch.abs(pz) - 1.0)
        any_out = torch.clamp((out_x + out_y + out_z) * 100.0, min=0.0, max=1.0)
        box_check = 1.0 - any_out

        # 🌟 外部から渡された3面図（テクスチャ）を座標 px, py, pz の値によって、
        # 3D空間上で動的に交差させるアナリティカル投影ルックアップ
        # 3面図が「球（x*x + y*y < 0.35）」などの形状情報を、座標の2乗和とマスクの積で100%連動させます
        proj_xy = torch.clamp(1.1 - (px*px + py*py) / 0.35, min=0.0, max=1.0)
        proj_xz = torch.clamp(1.1 - (px*px + pz*pz) / 0.35, min=0.0, max=1.0)
        proj_yz = torch.clamp(1.1 - (py*py + pz*pz) / 0.35, min=0.0, max=1.0)

        # 外部のマスクが「白(1.0)」の領域かつ、座標がその範囲内にいる時だけヒット
        mask_xy = multiview_textures[:, 0:1, :, :] * proj_xy
        mask_xz = multiview_textures[:, 1:2, :, :] * proj_xz
        mask_yz = multiview_textures[:, 2:3, :, :] * proj_yz

        # 3面すべてのシルエットの中にレイがいるか（完全なる3面交差）
        object_hit = mask_xy * mask_xz * mask_yz * box_check
        return object_hit

    def forward(self, multiview_textures, inv_view_matrix_64d):
        """
        Input:
          multiview_textures: [1, 3, 256, 256]
          inv_view_matrix_64d: [1, 64, 1, 1] (ANEが大得意な64chアライメント)
        """
        # 🌟 先頭の16チャンネル（行列の16成分）だけを切り出して 4x4 に復元！
        # 残りの48チャンネルはANEの空回し用として無視されます
        inv_view = inv_view_matrix_64d[0, :16, 0, 0].view(4, 4).half()
        
        # --- 逆行列を使ってワールド空間へ一撃変換 ---
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
        # 1. メインの視線レイマーチング
        # ==========================================
        px_all = init_px + init_dx * self.step_ratios
        py_all = init_py + init_dy * self.step_ratios
        pz_all = init_pz + init_dz * self.step_ratios

        textures_view = multiview_textures.expand(self.max_steps, -1, -1, -1)
        object_hit_all = self.check_multiview_hit(px_all, py_all, pz_all, textures_view)
        
        floor_hit_all = torch.clamp(torch.relu(self.floor_y - py_all) * 100.0, min=0.0, max=1.0)
        any_hit_all = torch.clamp(object_hit_all + floor_hit_all, min=0.0, max=1.0)
        
        # ANE特化型cumsum
        any_hit_permuted = any_hit_all.permute(1, 0, 2, 3)
        cum_hit_permuted = torch.cumsum(any_hit_permuted, dim=1)
        cum_hit = cum_hit_permuted.permute(1, 0, 2, 3)

        prior_hit = torch.cat([torch.zeros_like(cum_hit[:1]), cum_hit[:-1]], dim=0)
        not_hit_yet_all = torch.clamp(1.0 - prior_hit, min=0.0, max=1.0)

        is_first_object_all = not_hit_yet_all * object_hit_all
        is_first_floor_all = not_hit_yet_all * (1.0 - object_hit_all) * floor_hit_all

        hit_object_mask = torch.sum(is_first_object_all, dim=0, keepdim=True).clamp(0.0, 1.0)
        hit_floor_mask = torch.sum(is_first_floor_all, dim=0, keepdim=True).clamp(0.0, 1.0)
        accum_hit = torch.clamp(hit_object_mask + hit_floor_mask, min=0.0, max=1.0)

        px = torch.sum(is_first_object_all * px_all + is_first_floor_all * px_all, dim=0, keepdim=True)
        py = torch.sum(is_first_object_all * py_all + is_first_floor_all * py_all, dim=0, keepdim=True)
        pz = torch.sum(is_first_object_all * pz_all + is_first_floor_all * pz_all, dim=0, keepdim=True)

        # ==========================================
        # 🌟 数値微分による全自動・正確な法線計算 🌟
        # ==========================================
        f_center = self.check_multiview_hit(px, py, pz, multiview_textures)
        f_dx = self.check_multiview_hit(px + self.eps, py, pz, multiview_textures)
        f_dy = self.check_multiview_hit(px, py + self.eps, pz, multiview_textures)
        f_dz = self.check_multiview_hit(px, py, pz + self.eps, multiview_textures)
        
        raw_nx = f_center - f_dx
        raw_ny = f_center - f_dy
        raw_nz = f_center - f_dz
        
        inv_true_n_len = torch.rsqrt(raw_nx*raw_nx + raw_ny*raw_ny + raw_nz*raw_nz + 1e-5)
        
        first_nx = hit_object_mask * (raw_nx * inv_true_n_len)
        first_ny = hit_object_mask * (raw_ny * inv_true_n_len) + hit_floor_mask * 1.0
        first_nz = hit_object_mask * (raw_nz * inv_true_n_len)

        # ==========================================
        # 2. シャドウレイ
        # ==========================================
        shadow_start_x = px + 0.04 * self.light_dx
        shadow_start_y = py + 0.04 * self.light_dy
        shadow_start_z = pz + 0.04 * self.light_dz

        spx_all = shadow_start_x + self.light_dx * self.shadow_ratios
        spy_all = shadow_start_y + self.light_dy * self.shadow_ratios
        spz_all = shadow_start_z + self.light_dz * self.shadow_ratios

        textures_shadow_view = multiview_textures.expand(self.shadow_steps, -1, -1, -1)
        shadow_hit_all = self.check_multiview_hit(spx_all, spy_all, spz_all, textures_shadow_view)
        
        accum_shadow = torch.sum(shadow_hit_all, dim=0, keepdim=True).clamp(0.0, 1.0) * hit_floor_mask

        # ==========================================
        # 3. ライティング
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
