import torch
import torch.nn as nn
import torch.nn.functional as F

class ANERayTracingCore(nn.Module):
    def __init__(self, width=256, height=256, max_steps=12):
        super().__init__()
        self.w = width
        self.h = height
        self.max_steps = max_steps
        self.dt = 0.1
        
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
        self.register_buffer("init_pz", torch.full((1, 1, self.h, self.w), 3.0).half()) # カメラをZ=3.0に配置
        
        # 【修正】入力3チャンネル(XYZ)をそのまま1チャンネルずつ通すための重み
        object_kernel = torch.zeros(3, 3, 1, 1)
        object_kernel[0, 0, 0, 0] = 1.0  # X
        object_kernel[1, 1, 0, 0] = 1.0  # Y
        object_kernel[2, 2, 0, 0] = 1.0  # Z
        self.register_buffer("object_kernel", object_kernel.half())
        
        self.register_buffer("radius_sq", torch.full((1, 1, 1, 1), 1.0).half()) # 半径1.0
        self.register_buffer("ONES", torch.ones(1, 1, self.h, self.w).half())

    def ray_march_step(self, px, py, pz, dx, dy, dz):
        # 1. レイを前進
        px = px + self.dt * dx
        py = py + self.dt * dy
        pz = pz + self.dt * dz
        
        # 2. 【バコン！】確実に同じ型・デバイスで結合
        p_concat = torch.cat([px, py, pz], dim=1).to(px.device).half()
        
        # 出力も3チャンネル [1, 3, H, W] になります
        v_to_center = F.conv2d(p_concat, self.object_kernel, bias=None)
        
        # 距離の二乗を計算 (中心0,0,0からの距離)
        dist_sq = (v_to_center[:, 0:1, :, :]**2 + 
                   v_to_center[:, 1:2, :, :]**2 + 
                   v_to_center[:, 2:3, :, :]**2)
        
        # 3. 当たり判定マスク (半径の二乗以下なら1.0)
        hit_mask = torch.clamp(torch.relu(self.radius_sq - dist_sq) * 1000.0, min=0.0, max=1.0)
        
        # 4. 法線ベクトルの計算
        nx = v_to_center[:, 0:1, :, :]
        ny = v_to_center[:, 1:2, :, :]
        nz = v_to_center[:, 2:3, :, :]
        inv_n_len = torch.rsqrt(nx*nx + ny*ny + nz*nz + 1e-5)
        nx, ny, nz = nx * inv_n_len, ny * inv_n_len, nz * inv_n_len
        
        # 5. 反射ベクトルの計算 (D_reflect = D - 2 * (D . N) * N)
        dot = dx * nx + dy * ny + dz * nz
        rx = dx - 2.0 * dot * nx
        ry = dy - 2.0 * dot * ny
        rz = dz - 2.0 * dot * nz
        
        # 6. Where代用の算術ブレンド
        not_hit_mask = self.ONES - hit_mask
        
        dx_next = hit_mask * rx + not_hit_mask * dx
        dy_next = hit_mask * ry + not_hit_mask * dy
        dz_next = hit_mask * rz + not_hit_mask * dz
        
        return px, py, pz, dx_next, dy_next, dz_next, hit_mask

    def forward(self):
        px, py, pz = self.init_px, self.init_py, self.init_pz
        dx, dy, dz = self.init_dx, self.init_dy, self.init_dz
        
        accum_hit = torch.zeros(1, 1, self.h, self.w, device=px.device, dtype=torch.float16)
        
        # 24ステップ前進させてスキャン
        for _ in range(self.max_steps):
            px, py, pz, dx, dy, dz, hit_mask = self.ray_march_step(px, py, pz, dx, dy, dz)
            accum_hit = torch.clamp(accum_hit + hit_mask, min=0.0, max=1.0)
            
        return accum_hit
