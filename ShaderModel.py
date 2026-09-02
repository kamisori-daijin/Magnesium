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
        
        y_grid = torch.linspace(1.0, -1.0, self.internal_h, dtype=torch.float16).view(1, 1, self.internal_h, 1)
        x_grid = torch.linspace(-1.0, 1.0, self.internal_w, dtype=torch.float16).view(1, 1, 1, self.internal_w)
        
        self.register_buffer("x_grid_64ch", x_grid.expand(1, 64, self.internal_h, self.internal_w).contiguous())
        self.register_buffer("y_grid_64ch", y_grid.expand(1, 64, self.internal_h, self.internal_w).contiguous())
        
        rgb_kernel = torch.zeros(4, 64, 1, 1, dtype=torch.float16)
        rgb_kernel[0:3, :, 0, 0] = 1.0
        self.register_buffer("rgb_kernel", rgb_kernel)
        
        z_mask_kernel = torch.zeros(4, 64, 1, 1, dtype=torch.float16)
        z_mask_kernel[0, :, 0, 0] = 1.0
        z_mask_kernel[1, :, 0, 0] = 1.0
        self.register_buffer("z_mask_kernel", z_mask_kernel)
        
        self.register_buffer("ONES_64CH", torch.ones(1, 64, 1, 1, dtype=torch.float16))
        
        # パキッとMetal互換のZテストにするため、必要に応じて10.0からさらに大きな値（例: 1000.0）に調整してください
        self.register_buffer("SHARPNESS", torch.full((1, 64, 1, 1), 10.0, dtype=torch.float16))

    def forward(self, 
                A0, B0, C0, A1, B1, C1, A2, B2, C2, 
                R0, G0, B0_col, R1, G1, B1_col, R2, G2, B2_col,
                p0_iz, p1_iz, p2_iz,
                U0, V0, U1, V1, U2, V2,
                processed_texture):
        
        # Reshape [1, 64, 1, 1] 
        A0, B0, C0 = A0.view(1, 64, 1, 1), B0.view(1, 64, 1, 1), C0.view(1, 64, 1, 1)
        A1, B1, C1 = A1.view(1, 64, 1, 1), B1.view(1, 64, 1, 1), C1.view(1, 64, 1, 1)
        A2, B2, C2 = A2.view(1, 64, 1, 1), B2.view(1, 64, 1, 1), C2.view(1, 64, 1, 1)
        
        p0_iz, p1_iz, p2_iz = p0_iz.view(1, 64, 1, 1), p1_iz.view(1, 64, 1, 1), p2_iz.view(1, 64, 1, 1)
        U0, V0, U1, V1, U2, V2 = U0.view(1, 64, 1, 1), V0.view(1, 64, 1, 1), U1.view(1, 64, 1, 1), V1.view(1, 64, 1, 1), U2.view(1, 64, 1, 1), V2.view(1, 64, 1, 1)

        edges0 = (A0 * self.x_grid_64ch) + (B0 * self.y_grid_64ch) + C0
        edges1 = (A1 * self.x_grid_64ch) + (B1 * self.y_grid_64ch) + C1
        edges2 = (A2 * self.x_grid_64ch) + (B2 * self.y_grid_64ch) + C2

        valid_mask = torch.clamp(torch.relu((A0 * A0 + B0 * B0) * 100.0), min=0.0, max=1.0)
        inside_cw = torch.relu(edges0 * 100.0) * torch.relu(edges1 * 100.0) * torch.relu(edges2 * 100.0)
        mask = torch.clamp(inside_cw, min=0.0, max=1.0) * valid_mask 

        total_area = torch.clamp(edges0 + edges1 + edges2, min=1e-5)
        
        inv_total_area = torch.reciprocal(total_area)
        w0 = edges1 * inv_total_area
        w1 = edges2 * inv_total_area
        w2 = edges0 * inv_total_area

        pixel_inv_z = (p0_iz * w0 + p1_iz * w1 + p2_iz * w2) * mask 

        u_gradient = (U0 * w0 + U1 * w1 + U2 * w2)
        v_gradient = (V0 * w0 + V1 * w1 + V2 * w2)
        
        safe_inv_z = torch.clamp(pixel_inv_z, min=1e-4)
        inv_z_reciprocal = torch.reciprocal(safe_inv_z)
        
        u_sampler = processed_texture * (u_gradient * inv_z_reciprocal)
        v_sampler = processed_texture * (v_gradient * inv_z_reciprocal)
        sampled_texture = torch.clamp((u_sampler + v_sampler) * 0.5, min=0.0, max=1.0)

        # -------------------------------------------------------------
        # 【修正後】最大値（一番手前）ベースのZバッファ処理
        # -------------------------------------------------------------
        max_inv_z_per_pixel = torch.max(pixel_inv_z, dim=1, keepdim=True)[0]

        # 自分が「一番手前のポリゴン」からどれだけ奥にいるかの正確な差分
        z_diff = torch.relu(max_inv_z_per_pixel - pixel_inv_z) 

        # z_diff が 0.0（一番手前）なら z_blend_weights は 1.0、奥にあれば一瞬で 0.0 に張り付く
        z_blend_weights = torch.clamp(self.ONES_64CH - (z_diff * self.SHARPNESS), min=0.0, max=1.0)
        z_mask = mask * z_blend_weights 

        rgb_out = F.conv2d(sampled_texture * z_mask, self.rgb_kernel, bias=None)
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
