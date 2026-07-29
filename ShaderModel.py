import torch
import torch.nn as nn
import torch.nn.functional as F

class ANE3DRenderer64PixelShuffle(nn.Module):
    def __init__(self, width=256, height=256):
        super().__init__()
        self.width = width
        self.height = height
        
        # 1. （64x64）
        
        self.low_w = width // 4   # 64
        self.low_h = height // 4  # 64
        
        #  [1, 3, 64, 64] 
        y_coords = torch.linspace(1.0, -1.0, self.low_h).view(1, 1, self.low_h, 1)
        x_coords = torch.linspace(-1.0, 1.0, self.low_w).view(1, 1, 1, self.low_w)
        self.register_buffer("pixel_coords", torch.cat([
            x_coords.expand(1, 1, self.low_h, self.low_w),
            y_coords.expand(1, 1, self.low_h, self.low_w),
            torch.ones(1, 1, self.low_h, self.low_w)
        ], dim=1))
        
        # 2. [1, 64, 64, 64] ➔ [1, 4, 256, 256] 
        # (4x4) to 4ch(R, G, B, mask)
        # ※ upscale_factor=4  64 ➔ 64 / (4*4) = 4 Channel
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=4)

    def forward(self, A0, B0, C0, A1, B1, C1, A2, B2, C2, R0, G0, B0_col, R1, G1, B1_col, R2, G2, B2_col, z_weight):
        # Input: [1, 1, 1, 64] 
        
        # Conv2d
        # [1, 64, 64, 64]
        def compute_edges(A, B, C):
            weight = torch.cat([A, B, C], dim=1).permute(3, 1, 0, 2).contiguous()
            return F.conv2d(self.pixel_coords, weight, bias=None)

        # 1. Edge Function
        edges0 = compute_edges(A0, B0, C0)
        edges1 = compute_edges(A1, B1, C1)
        edges2 = compute_edges(A2, B2, C2)

        # 2. Create Mask 
        valid_mask = torch.clamp(torch.relu((A0**2 + B0**2) * 100.0), min=0.0, max=1.0).permute(3, 1, 0, 2)
        inside_cw = torch.relu(edges0 * 100.0) * torch.relu(edges1 * 100.0) * torch.relu(edges2 * 100.0)
        inside_ccw = torch.relu(-edges0 * 100.0) * torch.relu(-edges1 * 100.0) * torch.relu(-edges2 * 100.0)
        mask_low = torch.clamp(torch.maximum(inside_cw, inside_ccw) * valid_mask, min=0.0, max=1.0)

        # 3. Barycentric Coordinates
        total_area = torch.clamp(edges0 + edges1 + edges2, min=1e-5)
        w0 = edges1 / total_area
        w1 = edges2 / total_area
        w2 = edges0 / total_area

        def interpolate_color(c0, c1, c2):
            C0_w = c0.permute(3, 1, 0, 2)
            C1_w = c1.permute(3, 1, 0, 2)
            C2_w = c2.permute(3, 1, 0, 2)
            return (w0 * C0_w + w1 * C1_w + w2 * C2_w) * mask_low

        R_low = interpolate_color(R0, R1, R2)
        G_low = interpolate_color(G0, G1, G2)
        B_low = interpolate_color(B0_col, B1_col, B2_col)

        # 4. Z Buffer
        w_low = z_weight.permute(3, 1, 0, 2)
        
        # [1, 64, 64, 64] 
        R_out_low = R_low * w_low
        G_out_low = G_low * w_low
        B_out_low = B_low * w_low
        mask_out_low = mask_low * w_low
        
        # -------------------------------------------------------------------------
        # 64ch × 64px × 64px to 4ch × 256px × 256px 
        # -------------------------------------------------------------------------
        R = self.pixel_shuffle(R_out_low)           #　Shape: [1, 4, 256, 256]
        G = self.pixel_shuffle(G_out_low)           #　Shape: [1, 4, 256, 256]
        B = self.pixel_shuffle(B_out_low)           # Shape: [1, 4, 256, 256]
        mask_w = self.pixel_shuffle(mask_out_low)   # Shape: [1, 4, 256, 256]
        
        #  [1, 4, 256, 256]
        return R, G, B, mask_w
