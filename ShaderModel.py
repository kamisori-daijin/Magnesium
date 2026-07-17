import torch
import torch.nn as nn
import torch.nn.functional as F

class ANE3DRenderer64(nn.Module):
    def __init__(self, width=256, height=256):
        super().__init__()
        self.width = width
        self.height = height
        
        y_coords = torch.linspace(1.0, -1.0, height).view(1, 1, height, 1)
        x_coords = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
        self.register_buffer("pixel_coords", torch.cat([
            x_coords.expand(1, 1, height, width),
            y_coords.expand(1, 1, height, width),
            torch.ones(1, 1, height, width)
        ], dim=1))

    def forward(self, A0, B0, C0, A1, B1, C1, A2, B2, C2, R0, G0, B0_col, R1, G1, B1_col, R2, G2, B2_col, z_weight):
        # 入力はすべて [1, 1, 1, 64] を想定
        def compute_edges(A, B, C):
            weight = torch.cat([A, B, C], dim=1).permute(3, 1, 0, 2).contiguous()
            return F.conv2d(self.pixel_coords, weight, bias=None)

        # 1. エッジ関数（ピクセルが三角形の内側にあるか）の計算
        edges0 = compute_edges(A0, B0, C0)
        edges1 = compute_edges(A1, B1, C1)
        edges2 = compute_edges(A2, B2, C2)

        # 2. マスクの生成（内側なら1、外側なら0）
        valid_mask = torch.clamp(torch.relu((A0**2 + B0**2) * 100.0), min=0.0, max=1.0).permute(3, 1, 0, 2)
        inside_cw = torch.relu(edges0 * 100.0) * torch.relu(edges1 * 100.0) * torch.relu(edges2 * 100.0)
        inside_ccw = torch.relu(-edges0 * 100.0) * torch.relu(-edges1 * 100.0) * torch.relu(-edges2 * 100.0)
        mask = torch.clamp(torch.maximum(inside_cw, inside_ccw) * valid_mask, min=0.0, max=1.0)

        # 3. 重心座標系（Barycentric Coordinates）による色の補間
        total_area = torch.clamp(edges0 + edges1 + edges2, min=1e-5)
        w0 = edges1 / total_area
        w1 = edges2 / total_area
        w2 = edges0 / total_area

        def interpolate_color(c0, c1, c2):
            C0_w = c0.permute(3, 1, 0, 2)
            C1_w = c1.permute(3, 1, 0, 2)
            C2_w = c2.permute(3, 1, 0, 2)
            return (w0 * C0_w + w1 * C1_w + w2 * C2_w) * mask

        R = interpolate_color(R0, R1, R2)
        G = interpolate_color(G0, G1, G2)
        B = interpolate_color(B0_col, B1_col, B2_col)

        # 4. Zバッファ（奥行き）の適用
        w = z_weight.permute(3, 1, 0, 2)
        
        # 出力: [1, 64, H, W] のカラー画像とZ値
        return R * w, G * w, B * w, mask * w