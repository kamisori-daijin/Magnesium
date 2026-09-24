import torch
import torch.nn as nn
import torch.nn.functional as F

class ANE3DRenderer64(nn.Module):
    def __init__(self, target_width=1024, target_height=1024):
        super().__init__()
        self.target_width = target_width
        self.target_height = target_height
        
        self.internal_w = 256
        self.internal_h = 256
        
        y_grid = torch.linspace(1.0, -1.0, self.internal_h).view(1, 1, self.internal_h, 1)
        x_grid = torch.linspace(-1.0, 1.0, self.internal_w).view(1, 1, 1, self.internal_w)
        
        self.register_buffer("x_grid_64ch", x_grid.expand(1, 64, self.internal_h, self.internal_w).contiguous())
        self.register_buffer("y_grid_64ch", y_grid.expand(1, 64, self.internal_h, self.internal_w).contiguous())
        
        rgb_kernel = torch.zeros(4, 64, 1, 1)
        rgb_kernel[0:3, :, 0, 0] = 1.0
        self.register_buffer("rgb_kernel", rgb_kernel)
        
        z_mask_kernel = torch.zeros(4, 64, 1, 1)
        z_mask_kernel[0, :, 0, 0] = 1.0
        z_mask_kernel[1, :, 0, 0] = 1.0
        self.register_buffer("z_mask_kernel", z_mask_kernel)
        
        self.register_buffer("ONES_64CH", torch.ones(1, 64, 1, 1))
        self.register_buffer("SHARPNESS", torch.full((1, 64, 1, 1), 100.0))

    def forward(self, 
                A0, B0, C0, A1, B1, C1, A2, B2, C2, 
                R0, G0, B0_col, R1, G1, B1_col, R2, G2, B2_col,
                p0_iz, p1_iz, p2_iz,
                U0, V0, U1, V1, U2, V2,
                N0, N1, N2, 
                light_dir,  
                processed_texture):
        
        edges0 = (A0 * self.x_grid_64ch) + (B0 * self.y_grid_64ch) + C0
        edges1 = (A1 * self.x_grid_64ch) + (B1 * self.y_grid_64ch) + C1
        edges2 = (A2 * self.x_grid_64ch) + (B2 * self.y_grid_64ch) + C2

        valid_mask = torch.clamp(torch.relu((A0 * A0 + B0 * B0) * 100.0), min=0.0, max=1.0)
        inside_cw = torch.relu(edges0 * 100.0) * torch.relu(edges1 * 100.0) * torch.relu(edges2 * 100.0)
        mask = torch.clamp(inside_cw, min=0.0, max=1.0) * valid_mask
     
        total_area = edges0 + edges1 + edges2
        # 元の除算の形に戻しました
        inv_total_area = 1.0 / (total_area + 1e-5)
        
        w0 = edges1 * inv_total_area
        w1 = edges2 * inv_total_area
        w2 = edges0 * inv_total_area

        pixel_inv_z = (p0_iz * w0 + p1_iz * w1 + p2_iz * w2) * mask 

        u_gradient = (U0 * w0 + U1 * w1 + U2 * w2)
        v_gradient = (V0 * w0 + V1 * w1 + V2 * w2)
        
        R_blend = (R0 * w0 + R1 * w1 + R2 * w2)
        G_blend = (G0 * w0 + G1 * w1 + G2 * w2)
        B_blend = (B0 * w0 + B1 * w1 + B2 * w2)
        
        # 法線の補間
        Nx = N0[..., 0:1] * w0 + N1[..., 0:1] * w1 + N2[..., 0:1] * w2
        Ny = N0[..., 1:2] * w0 + N1[..., 1:2] * w1 + N2[..., 1:2] * w2
        Nz = N0[..., 2:3] * w0 + N1[..., 2:3] * w1 + N2[..., 2:3] * w2
        
        # ANE最適化: 逆平方根(rsqrt)を使って正規化の除算を回避
        length_sq = Nx*Nx + Ny*Ny + Nz*Nz + 1e-8
        inv_length = torch.rsqrt(length_sq)
        Nx, Ny, Nz = Nx * inv_length, Ny * inv_length, Nz * inv_length
        
        # 内積計算
        diffuse = Nx * light_dir[:, 0:1, :, :] + Ny * light_dir[:, 1:2, :, :] + Nz * light_dir[:, 2:3, :, :]
        diffuse = torch.clamp(diffuse, min=0.0, max=1.0)
        
        intensity = 0.2 + 0.8 * diffuse
        
        safe_inv_z = torch.clamp(pixel_inv_z, min=1e-4)
        # こちらも同様に元の除算の形に戻します
        inv_z_reciprocal = 1.0 / safe_inv_z
        
        u_sampler = processed_texture * (u_gradient * inv_z_reciprocal)
        v_sampler = processed_texture * (v_gradient * inv_z_reciprocal)
        sampled_texture = torch.clamp((u_sampler + v_sampler) * 0.5, min=0.0, max=1.0)

        final_color = sampled_texture * (R_blend + G_blend + B_blend) * intensity

        max_inv_z, _ = torch.max(pixel_inv_z, dim=1, keepdim=True)
        
        z_diff = torch.relu(max_inv_z - pixel_inv_z) 
        z_blend_weights = torch.clamp(self.ONES_64CH - (z_diff * self.SHARPNESS), min=0.0, max=1.0)
        z_mask = mask * z_blend_weights 

        rgb_out = F.conv2d(final_color * z_mask, self.rgb_kernel, bias=None)
        R_low = rgb_out[:, 0:1, :, :]
        G_low = rgb_out[:, 1:2, :, :]
        B_low = rgb_out[:, 2:3, :, :]
        
        max_inv_z_low = F.conv2d(pixel_inv_z * z_blend_weights, self.z_mask_kernel, bias=None)[:, 0:1, :, :]
        mask_w_low = F.conv2d(z_mask, self.z_mask_kernel, bias=None)[:, 1:2, :, :]
        
        R = F.interpolate(R_low, size=(self.target_height, self.target_width), mode='bilinear', align_corners=False)
        G = F.interpolate(G_low, size=(self.target_height, self.target_width), mode='bilinear', align_corners=False)
        B = F.interpolate(B_low, size=(self.target_height, self.target_width), mode='bilinear', align_corners=False)
        mask_w = F.interpolate(mask_w_low, size=(self.target_height, self.target_width), mode='bilinear', align_corners=False)
        max_inv_z = F.interpolate(max_inv_z_low, size=(self.target_height, self.target_width), mode='bilinear', align_corners=False)
        
        return R, G, B, mask_w, max_inv_z