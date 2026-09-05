import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class ANERayTracingCore(nn.Module):
    def __init__(self, width=256, height=256, max_steps=60, shadow_steps=15):
        super().__init__()
        self.w = width
        self.h = height
        self.max_steps = max_steps
        self.shadow_steps = shadow_steps
        self.dt = 0.08  # ステップ幅
        
        # --- カメラレイ（方向・初期位置）の生成 ---
        y_grid = torch.linspace(1.0, -1.0, self.h).view(1, 1, self.h, 1)
        x_grid = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        
        dx = x_grid.expand(1, 1, self.h, self.w)
        dy = y_grid.expand(1, 1, self.h, self.w)
        dz = torch.full((1, 1, self.h, self.w), -1.0)
        
        inv_len = torch.rsqrt(dx*dx + dy*dy + dz*dz + 1e-5)
        self.register_buffer("init_dx", (dx * inv_len).half())
        self.register_buffer("init_dy", (dy * inv_len).half())
        self.register_buffer("init_dz", (dz * inv_len).half())
        
        self.register_buffer("init_px", torch.zeros(1, 1, self.h, self.w).half())
        self.register_buffer("init_py", torch.zeros(1, 1, self.h, self.w).half())
        self.register_buffer("init_pz", torch.full((1, 1, self.h, self.w), 2.5).half()) 
        
        # --- 各種定数・環境設定バッファ ---
        self.floor_y = -0.8
        self.register_buffer("ONES", torch.ones(1, 1, self.h, self.w).half())
        self.register_buffer("ZEROS", torch.zeros(1, 1, self.h, self.w).half())
        
        # 平行光源の方向（右上から照射）
        light_dir_x = torch.full((1, 1, 1, 1), 1.0)
        light_dir_y = torch.full((1, 1, 1, 1), 1.0)
        light_dir_z = torch.full((1, 1, 1, 1), 1.0)
        inv_l_len = torch.rsqrt(light_dir_x*light_dir_x + light_dir_y*light_dir_y + light_dir_z*light_dir_z + 1e-5)
        self.register_buffer("light_dx", (light_dir_x * inv_l_len).half())
        self.register_buffer("light_dy", (light_dir_y * inv_l_len).half())
        self.register_buffer("light_dz", (light_dir_z * inv_l_len).half())

        # モデル内部だけで使うステップ係数テーブルを事前登録 [Steps, 1, 1, 1]
        step_ratios = torch.arange(self.max_steps).view(self.max_steps, 1, 1, 1) * self.dt
        self.register_buffer("step_ratios", step_ratios.half())
        
        shadow_ratios = torch.arange(self.shadow_steps).view(self.shadow_steps, 1, 1, 1) * self.dt
        self.register_buffer("shadow_ratios", shadow_ratios.half())

    def check_multiview_hit(self, px, py, pz, multiview_textures):
        """
        [B, 1, H, W] に拡張された座標に対して一斉に判定を行う
        """
        # 外部から渡された3面図マスクをバラす
        mask_xy = multiview_textures[:, 0:1, :, :]  # 正面図 (X, Y)
        mask_xz = multiview_textures[:, 1:2, :, :]  # 真上図 (X, Z)
        mask_yz = multiview_textures[:, 2:3, :, :]  # 真横図 (Y, Z)

        # 3面交差判定
        object_hit = mask_xy * mask_xz * mask_yz
        
        # バウンディングボックス境界判定
        out_x = torch.relu(torch.abs(px) - 1.0)
        out_y = torch.relu(torch.abs(py) - 1.0)
        out_z = torch.relu(torch.abs(pz) - 1.0)

        any_out = torch.clamp((out_x + out_y + out_z) * 1000.0, min=0.0, max=1.0)
        box_check = 1.0 - any_out
        
        return object_hit * box_check

    def forward(self, multiview_textures):
        """
        Input multiview_textures: [1, 3, 256, 256] 固定
        """
        # ==========================================
        # 1. メインの視線レイマーチング (内部バッチ = 60)
        # ==========================================
        # [60, 1, H, W] の空間位置を一撃で展開して計算
        px_all = self.init_px + self.init_dx * self.step_ratios
        py_all = self.init_py + self.init_dy * self.step_ratios
        pz_all = self.init_pz + self.init_dz * self.step_ratios

        # 3面図を入力の [1, 3, H, W] から [60, 3, H, W] にブロードキャスト
        textures_view = multiview_textures.expand(self.max_steps, -1, -1, -1)

        # オブジェクトのヒット判定を一斉スキャン [60, 1, H, W]
        object_hit_all = self.check_multiview_hit(px_all, py_all, pz_all, textures_view)
        
        # 床との交差判定 (Py が floor_y 以下)
        floor_hit_all = torch.clamp(torch.relu(self.floor_y - py_all) * 1000.0, min=0.0, max=1.0)
        
        # 「いずれかにヒットした」マスク
        any_hit_all = torch.clamp(object_hit_all + floor_hit_all, min=0.0, max=1.0)
        
        # 累積和（cumsum）を使って、過去のステップで既にヒットしているかフラグを再現
        cum_hit = torch.cumsum(any_hit_all, dim=0)
        prior_hit = torch.cat([torch.zeros_like(cum_hit[:1]), cum_hit[:-1]], dim=0)
        not_hit_yet_all = torch.clamp(1.0 - prior_hit, min=0.0, max=1.0)

        # 「そのステップで初めて」ヒットした箇所のマスク
        is_first_object_all = not_hit_yet_all * object_hit_all
        is_first_floor_all = not_hit_yet_all * (1.0 - object_hit_all) * floor_hit_all

        # 全ステップを統合して元の [1, 1, H, W] に戻す
        hit_object_mask = torch.sum(is_first_object_all, dim=0, keepdim=True).clamp(0.0, 1.0)
        hit_floor_mask = torch.sum(is_first_floor_all, dim=0, keepdim=True).clamp(0.0, 1.0)
        accum_hit = torch.clamp(hit_object_mask + hit_floor_mask, min=0.0, max=1.0)

        # 最終衝突位置の抽出 [1, 1, H, W]
        px = torch.sum(is_first_object_all * px_all + is_first_floor_all * px_all, dim=0, keepdim=True)
        py = torch.sum(is_first_object_all * py_all + is_first_floor_all * py_all, dim=0, keepdim=True)
        pz = torch.sum(is_first_object_all * pz_all + is_first_floor_all * pz_all, dim=0, keepdim=True)

        # 簡易法線計算
        inv_n_len = torch.rsqrt(px*px + py*py + pz*pz + 1e-5)
        obj_nx, obj_ny, obj_nz = px * inv_n_len, py * inv_n_len, pz * inv_n_len
        
        first_nx = hit_object_mask * obj_nx
        first_ny = hit_object_mask * obj_ny + hit_floor_mask * 1.0
        first_nz = hit_object_mask * obj_nz

        # ==========================================
        # 2. シャドウレイの並列化 (内部バッチ = 15)
        # ==========================================
        # 床に当たったピクセルからのみ、ライト方向へ進むレイの座標を一斉生成
        shadow_start_x = px + 0.04 * self.light_dx
        shadow_start_y = py + 0.04 * self.light_dy
        shadow_start_z = pz + 0.04 * self.light_dz

        # ここでシャドウ用の内部バッチ [15, 1, H, W] を一斉計算
        spx_all = shadow_start_x + self.light_dx * self.shadow_ratios
        spy_all = shadow_start_y + self.light_dy * self.shadow_ratios
        spz_all = shadow_start_z + self.light_dz * self.shadow_ratios

        # 3面図を入力の [1, 3, H, W] から [15, 3, H, W] にブロードキャスト
        textures_shadow_view = multiview_textures.expand(self.shadow_steps, -1, -1, -1)

        # ライトへ向かう途中でオブジェクトをかすめるか一斉スキャン [15, 1, H, W]
        shadow_hit_all = self.check_multiview_hit(spx_all, spy_all, spz_all, textures_shadow_view)
        
        # 1度でも遮られたら影にする [1, 1, H, W] に戻す
        accum_shadow = torch.sum(shadow_hit_all, dim=0, keepdim=True).clamp(0.0, 1.0)
        accum_shadow = accum_shadow * hit_floor_mask

        # ==========================================
        # 3. ライティング ＆ チェッカー床
        # ==========================================
        diffuse = first_nx * self.light_dx + first_ny * self.light_dy + first_nz * self.light_dz
        shading = torch.relu(diffuse) + 0.15
        
        sign_x = torch.clamp(px * 3.0 * 1000.0, min=-1.0, max=1.0)
        sign_z = torch.clamp(pz * 3.0 * 1000.0, min=-1.0, max=1.0)
        checker = (sign_x * sign_z + 1.0) * 0.5
        floor_color = hit_floor_mask * (0.3 + 0.2 * checker)
        
        base_color = hit_object_mask * self.ONES + floor_color
        light_modifier = (self.ONES - accum_shadow) * shading + accum_shadow * 0.15
        
        final_color = accum_hit * base_color * light_modifier
        return final_color
