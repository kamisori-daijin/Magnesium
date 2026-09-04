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
        self.max_steps = max_steps          # メインの視線レイの前進ステップ数
        self.shadow_steps = shadow_steps      # 影チェックレイの前進ステップ数
        self.dt = 0.1                       # レイの前進幅
        
        # --- カメラレイ（方向・初期位置）の生成 ---
        y_grid = torch.linspace(1.0, -1.0, self.h).view(1, 1, self.h, 1)
        x_grid = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        
        dx = x_grid.expand(1, 1, self.h, self.w)
        dy = y_grid.expand(1, 1, self.h, self.w)
        dz = torch.full((1, 1, self.h, self.w), -1.0)  # 奥へ進むレイ
        
        inv_len = torch.rsqrt(dx*dx + dy*dy + dz*dz + 1e-5)
        self.register_buffer("init_dx", (dx * inv_len).half())
        self.register_buffer("init_dy", (dy * inv_len).half())
        self.register_buffer("init_dz", (dz * inv_len).half())
        
        self.register_buffer("init_px", torch.zeros(1, 1, self.h, self.w).half())
        self.register_buffer("init_py", torch.zeros(1, 1, self.h, self.w).half())
        self.register_buffer("init_pz", torch.full((1, 1, self.h, self.w), 3.0).half()) # カメラをZ=3.0に配置
        
        # --- 各種定数・環境設定バッファ ---
        self.register_buffer("sphere_radius_sq", torch.full((1, 1, 1, 1), 1.0).half()) # 球体の半径^2 = 1.0
        self.floor_y = -1.0 # 床の高さ（Y = -1.0）
        
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

    def forward(self):
        px, py, pz = self.init_px, self.init_py, self.init_pz
        dx, dy, dz = self.init_dx, self.init_dy, self.init_dz
        
        # 状態管理マスク（Whereを使わない算術マスク処理用）
        accum_hit = self.ZEROS          # 何かに当たった全体のマスク
        hit_sphere_mask = self.ZEROS    # 球体に当たったマスク
        hit_floor_mask = self.ZEROS     # 床に当たったマスク
        
        # 最初の交点の法線記録用
        first_nx, first_ny, first_nz = self.ZEROS, self.ZEROS, self.ZEROS
        
        # ==========================================
        # STEP 1 & 2: メインの視線レイマーチング（60回アンロール）
        # ==========================================
        for _ in range(self.max_steps):
            # まだ何にも当たっていないピクセルだけ、レイを前進させる
            # ANEが最も得意とする乗算による「進行制御マスク」
            not_hit_yet = self.ONES - accum_hit
            px = px + self.dt * dx * not_hit_yet
            py = py + self.dt * dy * not_hit_yet
            pz = pz + self.dt * dz * not_hit_yet
            
            # ① 球体との交差判定（中心0,0,0からの距離の二乗）
            sphere_dist_sq = px*px + py*py + pz*pz
            sphere_hit = torch.clamp(torch.relu(self.sphere_radius_sq - sphere_dist_sq) * 1000.0, min=0.0, max=1.0)
            
            # ② 床との交差判定（Py が floor_y 以下になった瞬間ヒット）
            floor_hit = torch.clamp(torch.relu(self.floor_y - py) * 1000.0, min=0.0, max=1.0)
            
            # 【初ヒット判定の力技】今回新しく球体 / 床に当たったピクセルを計算
            is_first_sphere = not_hit_yet * sphere_hit
            # すでに球体に当たっている場所には、床の判定を上書きさせない
            is_first_floor = not_hit_yet * (self.ONES - sphere_hit) * floor_hit
            
            # それぞれの物体ごとの累積マスクを更新
            hit_sphere_mask = torch.clamp(hit_sphere_mask + is_first_sphere, min=0.0, max=1.0)
            hit_floor_mask = torch.clamp(hit_floor_mask + is_first_floor, min=0.0, max=1.0)
            accum_hit = torch.clamp(hit_sphere_mask + hit_floor_mask, min=0.0, max=1.0)
            
            # 法線の計算（球体なら中心からの方向、床なら真上[0,1,0]固定）
            inv_n_len = torch.rsqrt(sphere_dist_sq + 1e-5)
            sphere_nx, sphere_ny, sphere_nz = px * inv_n_len, py * inv_n_len, pz * inv_n_len
            
            # 算術ブレンドで法線を一斉に切り替えて累積
            first_nx = first_nx + is_first_sphere * sphere_nx + is_first_floor * 0.0
            first_ny = first_ny + is_first_sphere * sphere_ny + is_first_floor * 1.0  # 床の法線Yは1.0固定
            first_nz = first_nz + is_first_sphere * sphere_nz + is_first_floor * 0.0

        # ==========================================
        # STEP 3: 【脳汁コア】シャドウレイの脳筋アンロール
        # ==========================================
        # 「床に当たったピクセル」だけ、レイの向きを一斉に「ライトの方向」にパキッと書き換える！
        # これにより、床の表面からライトに向かって一斉に光線が逆走を始めます。
        dx = hit_floor_mask * self.light_dx + (self.ONES - hit_floor_mask) * dx
        dy = hit_floor_mask * self.light_dy + (self.ONES - hit_floor_mask) * dy
        dz = hit_floor_mask * self.light_dz + (self.ONES - hit_floor_mask) * dz
        
        # 影の遮蔽を記録する累積シャドウバッファ
        accum_shadow = self.ZEROS
        
        # 衝突点から少しだけ浮かせる（自己遮蔽によるジャギやノイズを防ぐための微小オフセット）
        px = px + 0.05 * dx
        py = py + 0.05 * dy
        pz = pz + 0.05 * dz
        
        # ライトへ向かって追加で15ステップ前進させて、球体をかすめるかチェック
        # これがForwardの後半に完全に一本のパイプラインとしてドッキングされます
        for _ in range(self.shadow_steps):
            px = px + self.dt * dx * hit_floor_mask
            py = py + self.dt * dy * hit_floor_mask
            pz = pz + self.dt * dz * hit_floor_mask
            
            # ライトへ向かう途中で、球体（中心0,0,0、半径1.0）の中を通過するかスキャン
            shadow_dist_sq = px*px + py*py + pz*pz
            in_shadow_mask = torch.clamp(torch.relu(self.sphere_radius_sq - shadow_dist_sq) * 1000.0, min=0.0, max=1.0)
            
            # 一度でも球体にぶつかったら、その床ピクセルの「影マスク」を1.0にする
            accum_shadow = torch.clamp(accum_shadow + in_shadow_mask, min=0.0, max=1.0)
            
        # 影マスクは「床のピクセル」だけに限定させる安全弁
        accum_shadow = accum_shadow * hit_floor_mask

        # ==========================================
        # ライティング ＆ 影の叩き潰し処理
        # ==========================================
        # ① 通常のディフューズ（環境光 0.15 込み）の計算
        diffuse = first_nx * self.light_dx + first_ny * self.light_dy + first_nz * self.light_dz
        shading = torch.relu(diffuse) + 0.15
        
        # 床に特有のチェッカーボード（格子模様）を数式だけで生成して見た目をリッチに！
        # 座標の正負を考慮したANE向けの算術パターン
        sign_x = torch.clamp(px * 1000.0, min=-1.0, max=1.0)
        sign_z = torch.clamp(pz * 1000.0, min=-1.0, max=1.0)
        
        # あとは同じように掛け算するだけ
        checker = (sign_x * sign_z + 1.0) * 0.5
        floor_color = hit_floor_mask * (0.3 + 0.2 * checker)
        
        # 物体の基本色（球体は白、床は格子模様）
        base_color = hit_sphere_mask * self.ONES + floor_color
        
        # ② 【影の叩き潰し】影マスク（accum_shadow）が1.0の場所だけ、光の強度を0.15（環境光のみ）に強制リセットする
        # これにより、床の上に球体の形をした美しい「本物のリアルな影」が焼き付きます！
        light_modifier = (self.ONES - accum_shadow) * shading + accum_shadow * 0.15
        
        # 最終カラー合成（背景は真っ黒）
        final_color = accum_hit * base_color * light_modifier
        
        return final_color
