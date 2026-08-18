import torch
import torch.nn as nn
import torch.nn.functional as F

class ANE3DRenderer64(nn.Module):
    def __init__(self, width=128, height=128):
        super().__init__()
        self.width = width
        self.height = height
        
     
        y_base = torch.linspace(1.0, -1.0, self.height, dtype=torch.float16).view(1, 1, self.height, 1)
        x_base = torch.linspace(-1.0, 1.0, self.width, dtype=torch.float16).view(1, 1, 1, self.width)
        
        self.register_buffer("x_base", x_base)
        self.register_buffer("y_base", y_base)
        self.register_buffer("sum_kernel", torch.ones(1, 64, 1, 1, dtype=torch.float16))

    def forward(self, 
                A0, B0, C0, A1, B1, C1, A2, B2, C2, 
                R0, G0, B0_col, R1, G1, B1_col, R2, G2, B2_col,
                p0_iz, p1_iz, p2_iz,
                U0, V0, U1, V1, U2, V2,
                processed_texture,
                tile_offset_x, tile_offset_y):
        
        # Tile Offset (1, 1, 1, 1)
        tile_offset_x = tile_offset_x.squeeze().view(1, 1, 1, 1)
        tile_offset_y = tile_offset_y.squeeze().view(1, 1, 1, 1)
        
        pixel_x = self.x_base + tile_offset_x  # (1, 1, 128, 128)
        pixel_y = self.y_base - tile_offset_y  # (1, 1, 128, 128)
        
  
        a0 = A0.view(1, 64, 1, 1)
        b0 = B0.view(1, 64, 1, 1)
        c0 = C0.view(1, 64, 1, 1)
        
        a1 = A1.view(1, 64, 1, 1)
        b1 = B1.view(1, 64, 1, 1)
        c1 = C1.view(1, 64, 1, 1)
        
        a2 = A2.view(1, 64, 1, 1)
        b2 = B2.view(1, 64, 1, 1)
        c2 = C2.view(1, 64, 1, 1)

        # 1, 64, 128, 128
        edges0 = (pixel_x * a0) + (pixel_y * b0) + c0
        edges1 = (pixel_x * a1) + (pixel_y * b1) + c1
        edges2 = (pixel_x * a2) + (pixel_y * b2) + c2

        # 2. Mask
        valid_mask = torch.clamp(torch.relu((a0**2 + b0**2) * 100.0), min=0.0, max=1.0)
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

 
        # weight [1, 64, 1, 1] 
        sum_inv_z = F.conv2d(pixel_inv_z, self.sum_kernel, bias=None)
        z_diff = torch.relu(sum_inv_z - pixel_inv_z)

        sharpness = 10.0 
        z_blend_weights = torch.clamp(1.0 - (z_diff * sharpness), min=0.0, max=1.0)
        z_mask = mask * z_blend_weights

        R_full = sampled_texture * z_mask
        G_full = sampled_texture * z_mask
        B_full = sampled_texture * z_mask
        mask_full = z_mask
        
        # 1, 1, 128, 128
        R = F.conv2d(R_full, self.sum_kernel, bias=None)
        G = F.conv2d(G_full, self.sum_kernel, bias=None)
        B = F.conv2d(B_full, self.sum_kernel, bias=None)
        mask_w = F.conv2d(mask_full, self.sum_kernel, bias=None)
        
        max_inv_z = F.conv2d(pixel_inv_z * z_blend_weights, self.sum_kernel, bias=None)
        
        return R, G, B, mask_w, max_inv_z