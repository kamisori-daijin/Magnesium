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
        
        # 全体マスク計算用のカーネル
        self.register_buffer("sum_kernel_All", torch.ones(1, 64, 1, 1, dtype=torch.float16))

    def forward(self, 
                A0, B0, C0, A1, B1, C1, A2, B2, C2, 
                R0, G0, B0_col, R1, G1, B1_col, R2, G2, B2_col,
                p0_iz, p1_iz, p2_iz,
                U0, V0, U1, V1, U2, V2,
                processed_texture):
        
        def compute_edges(A, B, C):
            weight = torch.cat([A, B, C], dim=1).permute(3, 1, 0, 2).contiguous()
            return F.conv2d(self.pixel_coords, weight, bias=None)

        edges0 = compute_edges(A0, B0, C0)
        edges1 = compute_edges(A1, B1, C1)
        edges2 = compute_edges(A2, B2, C2)

        valid_mask = torch.clamp(torch.relu((A0**2 + B0**2) * 100.0), min=0.0, max=1.0).permute(3, 1, 0, 2)
        inside_cw = torch.relu(edges0 * 100.0) * torch.relu(edges1 * 100.0) * torch.relu(edges2 * 100.0)
        mask = torch.clamp(inside_cw * valid_mask, min=0.0, max=1.0)

        total_area = torch.clamp(edges0 + edges1 + edges2, min=1e-5)
        inv_area = torch.pow(total_area, -1.0)
        
        w0 = edges1 * inv_area
        w1 = edges2 * inv_area
        w2 = edges0 * inv_area

        p0_z_space = p0_iz.view(1, 64, 1, 1) * w0
        p1_z_space = p1_iz.view(1, 64, 1, 1) * w1
        p2_z_space = p2_iz.view(1, 64, 1, 1) * w2
        pixel_inv_z = (p0_z_space + p1_z_space + p2_z_space) * mask

        u_gradient = (U0.view(1, 64, 1, 1) * w0 + U1.view(1, 64, 1, 1) * w1 + U2.view(1, 64, 1, 1) * w2)
        v_gradient = (V0.view(1, 64, 1, 1) * w0 + V1.view(1, 64, 1, 1) * w1 + V2.view(1, 64, 1, 1) * w2)
        
        u_sampler = processed_texture * u_gradient
        v_sampler = processed_texture * v_gradient
        sampled_texture = torch.clamp((u_sampler + v_sampler) * 0.5, min=0.0, max=1.0)

        sum_inv_z = F.conv2d(pixel_inv_z, self.sum_kernel_All, bias=None)
        z_diff = torch.relu(sum_inv_z - pixel_inv_z)

        sharpness = 10.0 
        z_blend_weights = torch.clamp(1.0 - (z_diff * sharpness), min=0.0, max=1.0)
        z_mask = mask * z_blend_weights

        # 💡 カラー合成: 0, 1, 2チャンネルからRGBを抽出
        full_color = sampled_texture * z_mask
        
        R = full_color[:, 0:1, :, :]
        G = full_color[:, 1:2, :, :]
        B = full_color[:, 2:3, :, :]
        
        mask_w = F.conv2d(z_mask, self.sum_kernel_All, bias=None)
        max_inv_z = F.conv2d(pixel_inv_z * z_blend_weights, self.sum_kernel_All, bias=None)
        
        return R, G, B, mask_w, max_inv_z