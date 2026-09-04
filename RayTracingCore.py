import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

class ANERayTracingCore(nn.Module):
    def __init__(self, width=256, height=256, max_steps=60):
        super().__init__()
        self.w = width
        self.h = height
        self.max_steps = max_steps
        self.dt = 0.1  # レイの1ステップの前進幅
        
        # --- カメラレイ（方向・初期位置）の生成 ---
        # ANEが最も安定する4次元 [1, C, H, W] 形状で、全ピクセル分のレイを最初から保持します
        y_grid = torch.linspace(1.0, -1.0, self.h).view(1, 1, self.h, 1)
        x_grid = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        
        # 初期方向ベクトル D (Dx, Dy, Dz)
        dx = x_grid.expand(1, 1, self.h, self.w)
        dy = y_grid.expand(1, 1, self.h, self.w)
        dz = torch.full((1, 1, self.h, self.w), -1.0)  # 奥（カメラの視線方向）へ進む
        
        # 幾何学的なL2正規化（方向ベクトルの長さを1にする）を rsqrt の力技で再現
        inv_len = torch.rsqrt(dx*dx + dy*dy + dz*dz + 1e-5)
        self.register_buffer("init_dx", (dx * inv_len).half())
        self.register_buffer("init_dy", (dy * inv_len).half())
        self.register_buffer("init_dz", (dz * inv_len).half())
        
        # 初期位置ベクトル P (Px, Py, Pz) -> カメラ位置
        self.register_buffer("init_px", torch.zeros(1, 1, self.h, self.w).half())
        self.register_buffer("init_py", torch.zeros(1, 1, self.h, self.w).half())
        self.register_buffer("init_pz", torch.full((1, 1, self.h, self.w), 3.0).half()) # カメラをZ=3.0（手前）に配置
        
        # --- 各種定数バッファの登録 ---
        self.register_buffer("radius_sq", torch.full((1, 1, 1, 1), 1.0).half()) # 球体の半径の二乗（1.0）
        self.register_buffer("ONES", torch.ones(1, 1, self.h, self.w).half())
        self.register_buffer("ZEROS", torch.zeros(1, 1, self.h, self.w).half())
        
        # 平行光源の方向（右上から手前に向けて光を飛ばす設定）
        light_dir_x = torch.full((1, 1, 1, 1), 1.0)
        light_dir_y = torch.full((1, 1, 1, 1), 1.0)
        light_dir_z = torch.full((1, 1, 1, 1), 1.0)
        inv_l_len = torch.rsqrt(light_dir_x*light_dir_x + light_dir_y*light_dir_y + light_dir_z*light_dir_z + 1e-5)
        self.register_buffer("light_dx", (light_dir_x * inv_l_len).half())
        self.register_buffer("light_dy", (light_dir_y * inv_l_len).half())
        self.register_buffer("light_dz", (light_dir_z * inv_l_len).half())

    def ray_march_step(self, px, py, pz, dx, dy, dz):
        """
        1ステップ分、全ピクセルのレイを一斉に直進させ、当たったら反射させるコア関数
        """
        # 1. レイを直進 (P = P + dt * D)
        px = px + self.dt * dx
        py = py + self.dt * dy
        pz = pz + self.dt * dz
        
        # 2. 地雷オペレータ `**2` を完全排除した、愚直な掛け算による距離の二乗計算 (x*x + y*y + z*z)
        # 中心が(0,0,0)の球体なので、現在の座標のままANEの積和演算（FMA）で超高速に解けます
        dist_sq = px*px + py*py + pz*pz
        
        # 3. 当たり判定マスクの生成（半径の二乗以下なら1.0）
        # ANE上で安全に動くよう、ReLUとクランプで 0.0 or 1.0 のマスクに変換します
        hit_mask = torch.clamp(torch.relu(self.radius_sq - dist_sq) * 1000.0, min=0.0, max=1.0)
        
        # 4. 法線ベクトルの計算 (球体の中心からの方向ベクトルを、幾何学的なL2正規化で抽出)
        inv_n_len = torch.rsqrt(dist_sq + 1e-5)
        nx = px * inv_n_len
        ny = py * inv_n_len
        nz = pz * inv_n_len
        
        # 5. 反射ベクトルの計算 (D_reflect = D - 2 * (D . N) * N)
        dot = dx * nx + dy * ny + dz * nz
        rx = dx - 2.0 * dot * nx
        ry = dy - 2.0 * dot * ny
        rz = dz - 2.0 * dot * nz
        
        # 6. 地雷 `torch.where` を代用する「算術ブレンド」の力技
        # Output = mask * A + (1.0 - mask) * B
        not_hit_mask = self.ONES - hit_mask
        dx_next = hit_mask * rx + not_hit_mask * dx
        dy_next = hit_mask * ry + not_hit_mask * dy
        dz_next = hit_mask * rz + not_hit_mask * dz
        
        return px, py, pz, dx_next, dy_next, dz_next, hit_mask, nx, ny, nz

    def forward(self):
        # バッファから初期状態のレイをロード
        px, py, pz = self.init_px, self.init_py, self.init_pz
        dx, dy, dz = self.init_dx, self.init_dy, self.init_dz
        
        # 各種状態追跡用バッファ（すべて1ノックで更新可能）
        accum_hit = self.ZEROS
        first_nx = self.ZEROS
        first_ny = self.ZEROS
        first_nz = self.ZEROS
        
        # ANE（CoreML）を動かすための「グラフのアンロール（パイプライン静的展開）」
        # このfor文がCoreMLコンパイラによって、シリコン上に「1本の巨大な超高速回路」として焼き付けられます
        for _ in range(self.max_steps):
            px, py, pz, dx, dy, dz, hit_mask, nx, ny, nz = self.ray_march_step(px, py, pz, dx, dy, dz)
            
            # 【初ヒット判定の力技】一度も当たっていない（accum_hitが0）の場所だけ、法線を記録するマスク
            is_first_hit = torch.clamp((self.ONES - accum_hit) * hit_mask, min=0.0, max=1.0)
            
            first_nx = first_nx + is_first_hit * nx
            first_ny = first_ny + is_first_hit * ny
            first_nz = first_nz + is_first_hit * nz
            
            # 全体のヒットマスクを累積
            accum_hit = torch.clamp(accum_hit + hit_mask, min=0.0, max=1.0)
            
        # --- 物理ベース・ライティング層 (Lambertian Shading) ---
        # 記録した最初の交点の法線(N)と、ライトの方向(L)の内積（ドット積）を計算
        # ランバート余弦則：コサイン項がそのまま陰影のグラデーションになります
        diffuse = first_nx * self.light_dx + first_ny * self.light_dy + first_nz * self.light_dz
        
        # 光が当たっていない裏側（マイナス値）を0.0にクランプし、少しの環境光（0.15）を足してリッチに
        shading = torch.relu(diffuse) + 0.15
        
        # 最終的なカラー出力（球体がある場所だけ陰影を適用し、背景は真っ黒にする）
        final_color = accum_hit * shading
        
        return final_color