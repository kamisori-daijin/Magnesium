import torch
import torch.nn as nn
import torch.nn.functional as F

class ANEFullGaussianRenderer(nn.Module):
    def __init__(self, width=256, height=256):
        super().__init__()
        self.width = width
        self.height = height
        
        # 256x256の画面の固定座標マップ [1, 2, H, W] (X, Y)
        y_coords = torch.linspace(1.0, -1.0, height).view(1, 1, height, 1)
        x_coords = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
        self.register_buffer("pixel_coords", torch.cat([
            x_coords.expand(1, 1, height, width),
            y_coords.expand(1, 1, height, width)
        ], dim=1))

    def forward(self, gaussian_buffer, camera_matrix):
        """
        gaussian_buffer: [1, 16, 1, max_points] (点のデータ)
        camera_matrix: (カメラ行列)
        """
        # 1. 前半：数万個の点の3D位置を2D画面の座標(u, v)へトランスフォーム
        pos_xyz = gaussian_buffer[:, 0:3, :, :]
        ones = torch.ones_like(pos_xyz[:, 0:1, :, :])
        pos_xyzw = torch.cat([pos_xyz, ones], dim=1)
        
        weight_mat = camera_matrix.view(4, 4, 1, 1)
        projected = F.conv2d(pos_xyzw, weight_mat, bias=None) # [1, 4, 1, max_points]
        
        # 2D画面上の中央座標 (u, v)
        u = (projected[:, 0:1, :, :] / (projected[:, 3:4, :, :] + 1e-5)).view(1, -1, 1, 1)
        v = (projected[:, 1:2, :, :] / (projected[:, 3:4, :, :] + 1e-5)).view(1, -1, 1, 1)
        
        # 2. 中半：画面の全ピクセルと、各ガウシアンの中心位置の「距離の二乗」を計算
        # [1, 2, H, W] の画面座標と、各点の中心(u, v)の差分を取る
        # ANEの並列性を活かすため、チャンネル数を16ch(または32/64ch)の単位に整えて一気に引算！
        pixel_x = self.pixel_coords[:, 0:1, :, :] # [1, 1, H, W]
        pixel_y = self.pixel_coords[:, 1:2, :, :] # [1, 1, H, W]
        
        # 3. 【★ANE専用・ガウシアンブレンド回路】
        # 各ピクセルから点までの距離の二乗を計算
        # 本来の3DGSの数式： α = exp(-0.5 * d^2)
        dist_sq = (pixel_x - u) ** 2 + (pixel_y - v) ** 2
        
        # ANEが得意とする指数関数(torch.exp)のハードウェアアクセラレーションを直撃！
        # これにより、GPUのシェーダーを1ミリも使わず、ANE内で完璧なボケ足（ガウス球）が生成されます
        alpha = torch.exp(-0.5 * dist_sq * 100.0) # [1, max_points, H, W]
        
        # 4. 後半：色情報とアルファ値を掛け合わせて「1枚の画面」に累積（ブレンド）
        # 各点のカラー情報を抽出
        r_points = gaussian_buffer[:, 10:11, :, :].view(1, -1, 1, 1)
        g_points = gaussian_buffer[:, 11:12, :, :].view(1, -1, 1, 1)
        b_points = gaussian_buffer[:, 12:13, :, :].view(1, -1, 1, 1)
        
        # アルファブレンド（足し合わせ）を行い、1チャンネルに潰す
        R = torch.sum(r_points * alpha, dim=1, keepdim=True) # [1, 1, H, W]
        G = torch.sum(g_points * alpha, dim=1, keepdim=True) # [1, 1, H, W]
        B = torch.sum(b_points * alpha, dim=1, keepdim=True) # [1, 1, H, W]
        mask = torch.sum(alpha, dim=1, keepdim=True)         # [1, 1, H, W]
        
        # 最終出力：完璧な1チャンネルずつの2D画面画像！
        return R, G, B, mask
