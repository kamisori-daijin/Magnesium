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
        
        self.register_buffer("sum_kernel", torch.ones(1, 64, 1, 1, dtype=torch.float16))

    def forward(self, A0, B0, C0, A1, B1, C1, A2, B2, C2, R0, G0, B0_col, R1, G1, B1_col, R2, G2, B2_col, z_weight,
                processed_texture):
        
        def compute_edges(A, B, C):
            weight = torch.cat([A, B, C], dim=1).permute(3, 1, 0, 2).contiguous()
            return F.conv2d(self.pixel_coords, weight, bias=None)

        # 1. Rasterization
        edges0 = compute_edges(A0, B0, C0)
        edges1 = compute_edges(A1, B1, C1)
        edges2 = compute_edges(A2, B2, C2)

        valid_mask = torch.clamp(torch.relu((A0**2 + B0**2) * 100.0), min=0.0, max=1.0).permute(3, 1, 0, 2)
        inside_cw = torch.relu(edges0 * 100.0) * torch.relu(edges1 * 100.0) * torch.relu(edges2 * 100.0)
        
        
        mask = torch.clamp(inside_cw * valid_mask, min=0.0, max=1.0)

        total_area = torch.clamp(edges0 + edges1 + edges2, min=1e-5)
        w1 = edges2 / total_area
        w2 = edges0 / total_area

    
        u_sampler = processed_texture * w1 * 2.0
        v_sampler = processed_texture * (1.0 - w2) * 2.0
        
        # 0.0 〜 1.0 Clamp
        sampled_texture = torch.clamp((u_sampler + v_sampler) * 0.5, min=0.0, max=1.0)

        # abs
        safe_z_weight = torch.abs(z_weight).permute(3, 1, 0, 2)

        # RGB,Mask
        R_full = sampled_texture * safe_z_weight * mask
        G_full = sampled_texture * safe_z_weight * mask
        B_full = sampled_texture * safe_z_weight * mask
        mask_full = mask * safe_z_weight
        
        # Output 1x1 Conv
        R = F.conv2d(R_full, self.sum_kernel, bias=None)
        G = F.conv2d(G_full, self.sum_kernel, bias=None)
        B = F.conv2d(B_full, self.sum_kernel, bias=None)
        mask_w = F.conv2d(mask_full, self.sum_kernel, bias=None)
        
        return R, G, B, mask_w
