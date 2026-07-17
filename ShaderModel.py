import torch
import torch.nn as nn
import torch.nn.functional as F

class ANE3DRenderer(nn.Module):
    def __init__(self, max_triangles=21845, width=256, height=256):
        super().__init__()
        self.max_triangles = max_triangles
        self.width = width
        self.height = height
        
        # 画面のピクセル位置 [1, 3, H, W] をあらかじめ平坦化して [1, 3, H*W, 1] で登録！
        # これにより、実行時のreshapeを完全にこの世から消滅させます
        y_coords = torch.linspace(1.0, -1.0, height).view(1, 1, height, 1)
        x_coords = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
        X = x_coords.expand(1, 1, height, width).reshape(1, 1, height * width, 1)
        Y = y_coords.expand(1, 1, height, width).reshape(1, 1, height * width, 1)
        One = torch.ones_like(X)
        
        self.register_buffer("pixel_X_flat", X)
        self.register_buffer("pixel_Y_flat", Y)

    def forward(self, transformed_vertices):
        """
        transformed_vertices: [1, 3, 1, MAX_VERTICES]
        """
        # =========================================================================
        # ステップA: 頂点バッファを三角形の並びへ抽出 (形状はすべて [1, 1, 1, max_triangles])
        # =========================================================================
        valid_len = self.max_triangles * 3
        v_buffer = transformed_vertices[:, :, :, :valid_len]
        
        # すべてお尻の「幅（W）」の次元のまま処理！ANEの上限（2048ch）に1ミリも触れません！
        p0_X = v_buffer[:, 0:1, :, 0::3]
        p0_Y = v_buffer[:, 1:2, :, 0::3]
        p1_X = v_buffer[:, 0:1, :, 1::3]
        p1_Y = v_buffer[:, 1:2, :, 1::3]
        p2_X = v_buffer[:, 0:1, :, 2::3]
        p2_Y = v_buffer[:, 1:2, :, 2::3]

        z0 = v_buffer[:, 2:3, :, 0::3]
        z1 = v_buffer[:, 2:3, :, 1::3]
        z2 = v_buffer[:, 2:3, :, 2::3]
        avg_z = (z0 + z1 + z2) / 3.0

        # =========================================================================
        # ステップB: 直線方程式（A, B, C）の生成 [1, 1, 1, max_triangles]
        # =========================================================================
        A0 = p0_Y - p1_Y
        B0 = p1_X - p0_X
        C0 = -(A0 * p0_X + B0 * p0_Y)

        A1 = p1_Y - p2_Y
        B1 = p2_X - p1_X
        C1 = -(A1 * p1_X + B1 * p1_Y)

        A2 = p2_Y - p0_Y
        B2 = p0_X - p2_X
        C2 = -(A2 * p0_X + B2 * p0_Y)

        # =========================================================================
        # ステップC: ★【Warning完全全滅ハック】1Dフラット空間での並列ブロードキャスト
        # =========================================================================
        # pixel_X_flat 形状: [1, 1, H*W, 1]
        # A0 形状:          [1, 1, 1,   max_triangles]
        # 掛け算した瞬間、自動的に [1, 1, H*W, max_triangles] の4Dテンソルがビュー変更なしで爆誕！
        # ANEが最も得意とする「1ch画像の純粋な行列積（マトリクスユニット）」に直撃します！
        edges0 = self.pixel_X_flat * A0 + self.pixel_Y_flat * B0 + C0
        edges1 = self.pixel_X_flat * A1 + self.pixel_Y_flat * B1 + C1
        edges2 = self.pixel_X_flat * A2 + self.pixel_Y_flat * B2 + C2

        # =========================================================================
        # ステップD: 両面描画・カリング無効化 (ReLU) [1, 1, H*W, max_triangles]
        # =========================================================================
        valid_poly_mask = torch.relu((A0**2 + B0**2) * 100.0)
        valid_poly_mask = torch.clamp(valid_poly_mask, min=0.0, max=1.0)

        inside_clockwise = torch.relu(edges0 * 100.0) * torch.relu(edges1 * 100.0) * torch.relu(edges2 * 100.0)
        inside_counter = torch.relu(-edges0 * 100.0) * torch.relu(-edges1 * 100.0) * torch.relu(-edges2 * 100.0)
        
        raw_mask = torch.maximum(inside_clockwise, inside_counter) * valid_poly_mask
        all_triangles_mask = torch.clamp(raw_mask, min=0.0, max=1.0)

        # =========================================================================
        # ステップE: Zバッファ ＆ 3Dフラットシェーディング
        # =========================================================================
        poly_normal_Z = torch.abs(A0 * B2 - B0 * A2)
        shading = torch.clamp(poly_normal_Z * 5.0, min=0.3, max=1.0)

        z_weight = torch.clamp(1.0 - (avg_z / 4.0), min=0.0, max=1.0)
        
        weighted_triangles = all_triangles_mask * z_weight * shading
        
        # ★【ココが真の終着点】お尻の「幅」の次元（dim=3）から一撃で最大値プレス！
        # 吐き出される形状は [1, 1, H*W, 1]。変形ロスが完全にゼロです
        poly_space_flat, _ = torch.max(weighted_triangles, dim=3, keepdim=True)
        
        # 最後の画面の 256x256 へ現像する、たった1回の正方形reshape
        # チャンネル上限を跨がない平面の変形なので、ANECコンパイラは120%警告を出さずに通します！
        poly_space = poly_space_flat.reshape(1, 1, self.height, self.width)

        # =========================================================================
        # ステップF: Unity風の「3Dグリッドの地面」の現像 (2Dのままストレート合成)
        # =========================================================================
        # 地面の展開用のピクセル位置をここで一時復元（背景なので低コスト）
        y_coords_2d = torch.linspace(1.0, -1.0, self.height).view(1, 1, self.height, 1)
        x_coords_2d = torch.linspace(-1.0, 1.0, self.width).view(1, 1, 1, self.width)
        pixel_X = x_coords_2d.expand(1, 1, self.height, self.width)
        pixel_Y = y_coords_2d.expand(1, 1, self.height, self.width)

        safe_Y = torch.clamp(-pixel_Y - 0.2, min=1e-5)
        floor_world_Z = 0.4 / safe_Y
        floor_world_X = pixel_X * floor_world_Z
        
        floor_sdf = torch.clamp(safe_Y * 4.0, max=0.2)
        grid_pattern = torch.relu(torch.sin(floor_world_X * 6.0) * torch.sin(floor_world_Z * 6.0))
        floor_textured = floor_sdf * (grid_pattern * 0.85 + 0.15)

        # =========================================================================
        # ステップG: カラー現像
        # =========================================================================
        poly_mask = torch.clamp(poly_space * 1000.0, min=0.0, max=1.0)
        
        poly_color = torch.tensor([0.9, 0.9, 0.9]).view(1, 3, 1, 1)
        grid_color_A = torch.tensor([0.1, 0.6, 0.4]).view(1, 3, 1, 1)
        grid_color_B = torch.tensor([0.15, 0.15, 0.15]).view(1, 3, 1, 1)
        
        floor_color = grid_color_A * grid_pattern + grid_color_B * (1.0 - grid_pattern)
        final_rgb = (poly_mask * poly_color * poly_space) + ((1.0 - poly_mask) * floor_color * floor_textured * 5.0)
        
        return torch.clamp(final_rgb, min=0.0, max=1.0)
