import torch
import torch.nn as nn

class ANE3DRenderer(nn.Module):
    def __init__(self, steps=128, width=256, height=256, num_features=64):
        super().__init__()
        self.steps = steps
        self.width = width
        self.height = height
        self.num_features = num_features # 空間を表現する基底の数
        
        # 1. 固定のスクリーン2D空間バッファ
        y_coords = torch.linspace(1.0, -1.0, height).view(1, 1, height, 1)
        x_coords = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
        X = x_coords.expand(1, 1, height, width)
        Y = y_coords.expand(1, 1, height, width)
        Z = torch.ones_like(X)
        W = torch.zeros_like(X)
        self.register_buffer("static_space", torch.cat([X, Y, Z, W], dim=1))

        # 2. 奥方向へのステップ
        self.register_buffer("z_steps", torch.linspace(0.1, 4.0, steps).view(steps, 1, 1, 1))

        # 3. 最終画面へ一撃プレスするグループ 1x1 Conv2d
        self.compress_z_conv = nn.Conv2d(
            in_channels=steps * 3, out_channels=3, kernel_size=1, groups=3, bias=False
        )
        with torch.no_grad():
            z_weights = torch.linspace(1.0, 0.05, steps).view(1, steps, 1, 1).repeat(3, 1, 1, 1)
            self.compress_z_conv.weight.copy_(z_weights)

    # ★【完全数値現像】カメラ行列と、「何でも描画できる形状ウェイト」だけを受け取る
    def forward(self, camera_matrix, shape_weights):
        # ─────────────────────────────────────────────────────────
        # ステップA: カメラ回転
        # ─────────────────────────────────────────────────────────
        space_reshaped = self.static_space.squeeze(0).permute(1, 2, 0)
        ray_dir = torch.einsum('ij,hwj->hwi', camera_matrix, space_reshaped).permute(2, 0, 1).unsqueeze(0)
        
        X_prime = ray_dir[:, 0:1, :, :] * self.z_steps
        Y_prime = ray_dir[:, 1:2, :, :] * self.z_steps
        Z_prime = ray_dir[:, 2:3, :, :] * self.z_steps

        # ─────────────────────────────────────────────────────────
        # ステップB: 空間の「基底テンソル」を自動生成 [steps, num_features, H, W]
        # ─────────────────────────────────────────────────────────
        # 条件分岐を一切排除するため、空間の多項式（X, Y, Z, X^2, Y^2, Z^2, X*Y...）を自動で並べます
        # ANEは要素ごとの掛け算（**2など）が超得意なので一瞬です
        # モデルのステップBのfeaturesリストにこれを追加するだけで表現力が化けます
        x_base = X_prime[:, 0:1, :, :]
        y_base = Y_prime[:, 0:1, :, :]
        z_base = Z_prime[:, 0:1, :, :]

        features = [
            x_base, y_base, z_base,
            x_base**2, y_base**2, z_base**2,
            # ★【周波数を変えたサイン・コサイン波を濃密に並べる】
            torch.sin(x_base * 2.0), torch.cos(x_base * 2.0),
            torch.sin(y_base * 2.0), torch.cos(y_base * 2.0),
            torch.sin(z_base * 2.0), torch.cos(z_base * 2.0),
            torch.sin(x_base * 8.0), torch.sin(y_base * 8.0), torch.sin(z_base * 8.0),
            # ★【絶対値のノイズ（マルチフラクタル基底）】
            torch.abs(torch.sin(x_base * 5.0)), 
            torch.abs(torch.sin(y_base * 5.0))
        ]
        # これで正真正銘、純粋な1チャンネルが17枚集まった「形状決定テンソル」になります！
        space_features = torch.cat(features, dim=1)

        # ─────────────────────────────────────────────────────────
        # ステップC: ★1x1 Convの代わりに einsum で「外から渡された数値」と掛け算！
        # ─────────────────────────────────────────────────────────
        # shape_weights の形状: [features] (外から渡すただの数値配列)
        # この1行だけで、球体、立方体、その他すべての形状への「現像」が走ります
        sphere_sdf = torch.einsum('f,sfhw->shw', shape_weights, space_features).unsqueeze(1)

        # ─── あとの床の模様やRGBブレンド、最後の1個のConv2dプレスは100%これまで通り ───
        floor_base = -0.4 - Y_prime
        floor_sdf = torch.clamp(floor_base, max=0.2)
        floor_world_Z = (Z_prime / (Y_prime - 1e-5)) * -0.4
        floor_world_X = (X_prime / (Y_prime - 1e-5)) * -0.4
        grid_pattern = torch.relu(torch.sin(floor_world_X * 3.0) * torch.sin(floor_world_Z * 3.0))
        floor_textured = floor_sdf * (grid_pattern * 0.85 + 0.15)

        sphere_mask = torch.relu(sphere_sdf) # ★ここでReLUに流す！
        floor_mask = torch.relu(floor_textured)
        
        sphere_color = torch.tensor([0.9, 0.2, 0.1]).view(1, 3, 1, 1)
        grid_color_A = torch.tensor([0.1, 0.7, 0.4]).view(1, 3, 1, 1)
        grid_color_B = torch.tensor([0.9, 0.9, 0.9]).view(1, 3, 1, 1)
        floor_color = grid_color_A * grid_pattern + grid_color_B * (1.0 - grid_pattern)
        
        valid_space_rgb = (sphere_mask * sphere_color) + (floor_mask * floor_color)
        space_channels = valid_space_rgb.permute(1, 0, 2, 3).reshape(1, 3 * self.steps, self.height, self.width)
        
        framebuffer = self.compress_z_conv(space_channels)
        return torch.clamp(framebuffer, min=0.0, max=1.0)
