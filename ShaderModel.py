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
        
        self.z_space_expander = nn.Conv2d(64, 64, kernel_size=1, groups=64, bias=None)
        with torch.no_grad():
            self.z_space_expander.weight.copy_(torch.ones(64, 1, 1, 1))

    def forward(self, A0, B0, C0, A1, B1, C1, A2, B2, C2, R0, G0, B0_col, R1, G1, B1_col, R2, G2, B2_col, z_weight,
                processed_texture):
        
        def compute_edges(A, B, C):
            weight = torch.cat([A, B, C], dim=1).permute(3, 1, 0, 2).contiguous()
            return F.conv2d(self.pixel_coords, weight, bias=None)

        edges0 = compute_edges(A0, B0, C0)
        edges1 = compute_edges(A1, B1, C1)
        edges2 = compute_edges(A2, B2, C2)

        valid_mask = torch.clamp(torch.relu((A0**2 + B0**2) * 100.0), min=0.0, max=1.0).permute(3, 1, 0, 2)
        inside_cw = torch.relu(edges0 * 100.0) * torch.relu(edges1 * 100.0) * torch.relu(edges2 * 100.0)
        mask = torch.clamp(inside_cw * valid_mask, min=0.0, max=1.0) #  [1, 64, 256, 256] 

        total_area = torch.clamp(edges0 + edges1 + edges2, min=1e-5)
        inv_area = torch.pow(total_area, -1.0)
        
        w1 = edges2 * inv_area
        w2 = edges0 * inv_area

        #  w1, w2 [1, 64, 256, 256]
        u_sampler = processed_texture * w1 * 2.0
        v_sampler = processed_texture * (1.0 - w2) * 2.0
        sampled_texture = torch.clamp((u_sampler + v_sampler) * 0.5, min=0.0, max=1.0)

        # [1, 64, 1, 1] 
        # z_weight [1, 1, 1, 64] ➔ [1, 64, 1, 1] 
        safe_z_weight = torch.abs(z_weight).view(1, 64, 1, 1)
        
        dummy_ones = torch.ones(1, 64, 256, 256, dtype=torch.float16, device=z_weight.device)
        z_weight_space = safe_z_weight * dummy_ones

        # Z Buffer
        pixel_inv_z = z_weight_space * mask  # [1, 64, 256, 256]

        # 1x1 Conv（Sum）
        sum_inv_z = F.conv2d(pixel_inv_z, self.sum_kernel, bias=None)  # [1, 1, 256, 256]

        
        z_diff = torch.relu(sum_inv_z - pixel_inv_z)  # [1, 64, 256, 256]

        # Clamp and Mul (Blur weight)
        sharpness = 10.0 
        z_blend_weights = torch.clamp(1.0 - (z_diff * sharpness), min=0.0, max=1.0)

        # Z mask
        z_mask = mask * z_blend_weights  # [1, 64, 256, 256]

        # [1, 64, 256, 256] 
        # Mul
        R_full = sampled_texture * z_mask
        G_full = sampled_texture * z_mask
        B_full = sampled_texture * z_mask
        mask_full = z_mask
        
        #　Output 1x1 Conv Sum 
        R = F.conv2d(R_full, self.sum_kernel, bias=None)
        G = F.conv2d(G_full, self.sum_kernel, bias=None)
        B = F.conv2d(B_full, self.sum_kernel, bias=None)
        mask_w = F.conv2d(mask_full, self.sum_kernel, bias=None)
        
     
        max_inv_z = F.conv2d(pixel_inv_z * z_blend_weights, self.sum_kernel, bias=None)
        
        return R, G, B, mask_w, max_inv_z
