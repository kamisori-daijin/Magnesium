import torch
import torch.nn as nn

class ANE3DRenderer(nn.Module):
    def __init__(self, steps=128, width=256, height=256):
        super().__init__()
        self.steps = steps
        self.width = width
        self.height = height
        
        # 1. スクリーン上の2Dピクセル位置を固定バッファとして登録
        y_coords = torch.linspace(1.0, -1.0, height).view(1, 1, height, 1)
        x_coords = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
        
        X = x_coords.expand(1, 1, height, width)
        Y = y_coords.expand(1, 1, height, width)
        Z = torch.ones_like(X)
        W = torch.zeros_like(X)
        
        static_space = torch.cat([X, Y, Z, W], dim=1) # [1, 4, H, W]
        self.register_buffer("static_space", static_space)

        # 2. 奥方向（Steps）の進捗率をバッファとして登録
        z_steps = torch.linspace(0.1, 4.0, steps).view(steps, 1, 1, 1)
        self.register_buffer("z_steps", z_steps)

        # 3. 2D画面へ潰す 1x1 Conv2d (RGBグループ化)
        self.compress_z_conv = nn.Conv2d(
            in_channels=steps * 3, out_channels=3, kernel_size=1, groups=3, bias=False
        )
        with torch.no_grad():
            z_weights = torch.linspace(1.0, 0.05, steps).view(1, steps, 1, 1)
            triple_weights = z_weights.repeat(3, 1, 1, 1)
            self.compress_z_conv.weight.copy_(triple_weights)

    # ★【完全動的化】カメラ行列とオブジェクトパラメータを2つの引数で受け取る
    def forward(self, camera_matrix, object_params):
        # =========================================================================
        # ステップA: 外部から入ってきたテンソルを使ってアインシュタイン和で回転
        # =========================================================================
        space_reshaped = self.static_space.squeeze(0).permute(1, 2, 0)
        ray_dir = torch.einsum('ij,hwj->hwi', camera_matrix, space_reshaped)
        ray_dir = ray_dir.permute(2, 0, 1).unsqueeze(0)
        
        X_prime = ray_dir[:, 0:1, :, :] * self.z_steps
        Y_prime = ray_dir[:, 1:2, :, :] * self.z_steps
        Z_prime = ray_dir[:, 2:3, :, :] * self.z_steps

        # =========================================================================
        # ステップB: 外部から渡された「動的オブジェクトデータ」の並列SDF評価
        # =========================================================================
        # object_params形状: [N, 4] -> 各行が [X, Y, Z, Radius]
        # ANEでブロードキャスト並列処理を行うために、各次元を [N, 1, 1, 1, 1] にバラす
        obj_X = object_params[:, 0].view(-1, 1, 1, 1, 1)
        obj_Y = object_params[:, 1].view(-1, 1, 1, 1, 1)
        obj_Z = object_params[:, 2].view(-1, 1, 1, 1, 1)
        obj_R = object_params[:, 3].view(-1, 1, 1, 1, 1)

        # 空間座標をN個分に自動引き伸ばしして距離の2乗を一撃計算
        dist_sq = (X_prime - obj_X)**2 + (Y_prime - obj_Y)**2 + (Z_prime - obj_Z)**2
        
        # 3つの球体マスクの強さをまとめて計算し、一番手前（最大値）をマージ
        # 形状: [N, steps, 1, H, W] -> [steps, 1, H, W]
        spheres_lighting = torch.relu(1.0 - (dist_sq / (obj_R**2 + 1e-5)))
        spheres_sdf = (obj_R**2 - dist_sq) * (spheres_lighting * 0.8 + 0.2)
        sphere_sdf, _ = torch.max(spheres_sdf, dim=0)
        
        # 2. 地面（床） Y=-0.4
        floor_base = -0.4 - Y_prime
        floor_sdf = torch.clamp(floor_base, max=0.2)

        # =========================================================================
        # ステップC〜E: 模様、カラーマスク、プレス（128ステップ対応版）
        # =========================================================================
        floor_world_Z = (Z_prime / (Y_prime - 1e-5)) * -0.4
        floor_world_X = (X_prime / (Y_prime - 1e-5)) * -0.4
        
        grid_pattern = torch.relu(torch.sin(floor_world_X * 3.0) * torch.sin(floor_world_Z * 3.0))
        floor_textured = floor_sdf * (grid_pattern * 0.85 + 0.15)

        sphere_mask = torch.relu(sphere_sdf)
        floor_mask = torch.relu(floor_textured)
        
        sphere_color = torch.tensor([0.9, 0.2, 0.1]).view(1, 3, 1, 1) # 赤
        grid_color_A = torch.tensor([0.1, 0.7, 0.4]).view(1, 3, 1, 1) # 緑
        grid_color_B = torch.tensor([0.9, 0.9, 0.9]).view(1, 3, 1, 1) # 白
        floor_color = grid_color_A * grid_pattern + grid_color_B * (1.0 - grid_pattern)
        
        valid_space_rgb = (sphere_mask * sphere_color) + (floor_mask * floor_color)

        permuted = valid_space_rgb.permute(1, 0, 2, 3)
        space_channels = permuted.reshape(1, 3 * self.steps, self.height, self.width)
        
        framebuffer = self.compress_z_conv(space_channels)
        return torch.clamp(framebuffer, min=0.0, max=1.0)
