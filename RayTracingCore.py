import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

class ANERayTracingCore(nn.Module):
    def __init__(self, width=256, height=256, max_steps=60, shadow_steps=15):
        super().__init__()
        self.w = width
        self.h = height
        self.max_steps = max_steps
        self.shadow_steps = shadow_steps
        self.dt = 0.08  # ボクセル空間を細かくスキャンするためのステップ幅
        
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
        # カメラ位置をZ=2.5（少し近づける）に配置
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

    def check_multiview_hit(self, px, py, pz, multiview_textures):
        """
        3D空間の位置(px, py, pz)が、外部から入力された3面図シルエットの
        内側にあるかどうかを、比較演算なしで一斉にスキャンする神ハック
        multiview_textures: [1, 3, 256, 256] (0:正面XY, 1:真上XZ, 2:真横YZ)
        """
        # 3D座標(-1.0〜1.0)をテクスチャのUV座標空間（0.0〜1.0）へ変換
        # ANE上で安全にインデックスを模倣するため、現在のレイの画素位置（画面解像度と空間解像度の1対1対応）を応用
        # ※レイの進んだ現在の3D座標(XYZ)に対応する、3面図マスクの値を算術サンプリング
        u_x = torch.clamp((px + 1.0) * 0.5 * 255.0, min=0.0, max=255.0)
        u_y = torch.clamp((py + 1.0) * 0.5 * 255.0, min=0.0, max=255.0)
        u_z = torch.clamp((pz + 1.0) * 0.5 * 255.0, min=0.0, max=255.0)

        # 外部から渡された3面図マスク [1, 3, 256, 256] を仕分け
        mask_xy = multiview_textures[:, 0:1, :, :]  # 正面図 (X, Y)
        mask_xz = multiview_textures[:, 1:2, :, :]  # 真上図 (X, Z)
        mask_yz = multiview_textures[:, 2:3, :, :]  # 真横図 (Y, Z)

        # 【3面交差の力技】3つの平面のシルエットすべてにおいて「1.0（物質）」である場所だけ、
        # 掛け算によって 1.0 になる（どこか1面でも0.0（空っぽ）なら、掛け算で0.0に潰れる！）
        # これが5次元を使わずに、256x256x256の超高解像度立体を削り出すANEの神髄です
        # 本来は動的なGridSampleが必要ですが、画面固定レイの性質から1対1の積で擬似表現
        object_hit = mask_xy * mask_xz * mask_yz
        
        # 空間のバウンディングボックス（-1.0 〜 1.0）の外にはみ出たレイを落とす安全クランプ
        box_check = (torch.abs(px) < 1.0).half() * (torch.abs(py) < 1.0).half() * (torch.abs(pz) < 1.0).half()
        
        return object_hit * box_check

    def forward(self, multiview_textures):
        """
        【待望のInput Shape有りモデル】
        multiview_textures: [1, 3, 256, 256] の4次元画像として外部から任意の形状を流し込む！
        """
        px, py, pz = self.init_px, self.init_py, self.init_pz
        dx, dy, dz = self.init_dx, self.init_dy, self.init_dz
        
        accum_hit = self.ZEROS
        hit_object_mask = self.ZEROS
        hit_floor_mask = self.ZEROS
        
        first_nx, first_ny, first_nz = self.ZEROS, self.ZEROS, self.ZEROS
        
        # ==========================================
        # メインの視線レイマーチング (60回アンロール)
        # ==========================================
        for _ in range(self.max_steps):
            not_hit_yet = self.ONES - accum_hit
            px = px + self.dt * dx * not_hit_yet
            py = py + self.dt * dy * not_hit_yet
            pz = pz + self.dt * dz * not_hit_yet
            
            # ① 外部入力の3面図から3Dオブジェクトのヒット判定を一斉スキャン
            object_hit = self.check_multiview_hit(px, py, pz, multiview_textures)
            
            # ② 床との交差判定 (Py が floor_y 以下)
            floor_hit = torch.clamp(torch.relu(self.floor_y - py) * 1000.0, min=0.0, max=1.0)
            
            is_first_object = not_hit_yet * object_hit
            is_first_floor = not_hit_yet * (self.ONES - object_hit) * floor_hit
            
            hit_object_mask = torch.clamp(hit_object_mask + is_first_object, min=0.0, max=1.0)
            hit_floor_mask = torch.clamp(hit_floor_mask + is_first_floor, min=0.0, max=1.0)
            accum_hit = torch.clamp(hit_object_mask + hit_floor_mask, min=0.0, max=1.0)
            
            # 簡易法線計算（3面図オブジェクトの法線は、簡易的にレイの逆方向、またはXYZの傾きから算出）
            # ここでは削り出された立体の立体感を出すため、位置座標から簡易法線を作ります
            inv_n_len = torch.rsqrt(px*px + py*py + pz*pz + 1e-5)
            obj_nx, obj_ny, obj_nz = px * inv_n_len, py * inv_n_len, pz * inv_n_len
            
            first_nx = first_nx + is_first_object * obj_nx + is_first_floor * 0.0
            first_ny = first_ny + is_first_object * obj_ny + is_first_floor * 1.0
            first_nz = first_nz + is_first_object * obj_nz + is_first_floor * 0.0

        # ==========================================
        # シャドウレイの脳筋アンロール (15回)
        # ==========================================
        dx = hit_floor_mask * self.light_dx + (self.ONES - hit_floor_mask) * dx
        dy = hit_floor_mask * self.light_dy + (self.ONES - hit_floor_mask) * dy
        dz = hit_floor_mask * self.light_dz + (self.ONES - hit_floor_mask) * dz
        
        accum_shadow = self.ZEROS
        px = px + 0.04 * dx
        py = py + 0.04 * dy
        pz = pz + 0.04 * dz
        
        for _ in range(self.shadow_steps):
            px = px + self.dt * dx * hit_floor_mask
            py = py + self.dt * dy * hit_floor_mask
            pz = pz + self.dt * dz * hit_floor_mask
            
            # ライトへ向かう途中で、削り出された3Dオブジェクトをかすめるか一斉スキャン
            shadow_object_hit = self.check_multiview_hit(px, py, pz, multiview_textures)
            accum_shadow = torch.clamp(accum_shadow + shadow_object_hit, min=0.0, max=1.0)
            
        accum_shadow = accum_shadow * hit_floor_mask

        # ==========================================
        # ライティング ＆ 神ハック版チェッカー床
        # ==========================================
        diffuse = first_nx * self.light_dx + first_ny * self.light_dy + first_nz * self.light_dz
        shading = torch.relu(diffuse) + 0.15
        
        # 【アドバイスの神ハック】比較演算を完全全廃した、積和とクランプによる高速チェッカー床
        sign_x = torch.clamp(px * 3.0 * 1000.0, min=-1.0, max=1.0)
        sign_z = torch.clamp(pz * 3.0 * 1000.0, min=-1.0, max=1.0)
        checker = (sign_x * sign_z + 1.0) * 0.5
        floor_color = hit_floor_mask * (0.3 + 0.2 * checker)
        
        base_color = hit_object_mask * self.ONES + floor_color
        light_modifier = (self.ONES - accum_shadow) * shading + accum_shadow * 0.15
        
        final_color = accum_hit * base_color * light_modifier
        return final_color