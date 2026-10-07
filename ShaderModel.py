import torch
import torch.nn as nn
import torch.nn.functional as F

class ANE3DRenderer(nn.Module):
    def __init__(self, target_width=1024, target_height=1024, max_polygons=256):
        super().__init__()
        self.target_width = target_width
        self.target_height = target_height
        
        self.internal_w = 256
        self.internal_h = 256
        self.max_polygons = max_polygons
        
        # [1, 1, H, W] の形状のまま保持し、メモリを節約
        y_grid = torch.linspace(1.0, -1.0, self.internal_h).view(1, 1, self.internal_h, 1)
        x_grid = torch.linspace(-1.0, 1.0, self.internal_w).view(1, 1, 1, self.internal_w)
        
        # expand せずにそのまま登録 (ブロードキャストは forward で行う)
        self.register_buffer("x_grid_ch", x_grid)
        self.register_buffer("y_grid_ch", y_grid)
        
        self.register_buffer("ONES_CH", torch.ones(1, self.max_polygons, 1, 1))

    def forward(self, 
                A0, B0, C0, A1, B1, C1, A2, B2, C2, 
                R0, G0, B0_col, R1, G1, B1_col, R2, G2, B2_col,
                p0_iz, p1_iz, p2_iz,
                U0, V0, U1, V1, U2, V2,
                processed_texture):
        
        # --- エッジ計算とマスク (Elementwise ALU) ---
        # x_grid_ch, y_grid_ch は [1, 1, H, W] なので、
        # [1, max_polygons, 1, 1] の A0 などと掛け合わせることで自動的にブロードキャストされる
        edges0 = (A0 * self.x_grid_ch) + (B0 * self.y_grid_ch) + C0
        edges1 = (A1 * self.x_grid_ch) + (B1 * self.y_grid_ch) + C1
        edges2 = (A2 * self.x_grid_ch) + (B2 * self.y_grid_ch) + C2

        valid_mask = torch.clamp(torch.relu((A0 * A0 + B0 * B0) * 100.0), min=0.0, max=1.0)
        inside_cw = torch.relu(edges0 * 100.0) * torch.relu(edges1 * 100.0) * torch.relu(edges2 * 100.0)
        mask = torch.clamp(inside_cw, min=0.0, max=1.0) * valid_mask
     
        # --- 重み補間 (Dedicated Reciprocal) ---
        total_area = edges0 + edges1 + edges2
        inv_total_area = 1.0 / (total_area + 1e-5)
        
        w0 = edges1 * inv_total_area
        w1 = edges2 * inv_total_area
        w2 = edges0 * inv_total_area

        pixel_inv_z = (p0_iz * w0 + p1_iz * w1 + p2_iz * w2) * mask 

        # --- カラー & UV補間 ---
        u_gradient = (U0 * w0 + U1 * w1 + U2 * w2)
        v_gradient = (V0 * w0 + V1 * w1 + V2 * w2)
        
        R_blend = (R0 * w0 + R1 * w1 + R2 * w2)
        G_blend = (G0 * w0 + G1 * w1 + G2 * w2)
        B_blend = (B0_col * w0 + B1_col * w1 + B2_col * w2)
        
        safe_inv_z = torch.clamp(pixel_inv_z, min=1e-4)
        inv_z_reciprocal = 1.0 / safe_inv_z
        
        u_sampler = processed_texture * (u_gradient * inv_z_reciprocal)
        v_sampler = processed_texture * (v_gradient * inv_z_reciprocal)
        sampled_texture = torch.clamp((u_sampler + v_sampler) * 0.5, min=0.0, max=1.0)

        final_R = sampled_texture * R_blend
        final_G = sampled_texture * G_blend
        final_B = sampled_texture * B_blend

        # --- ネイティブ Z-Buffer 処理 (Native Reductions) ---
        max_inv_z, _ = torch.max(pixel_inv_z, dim=1, keepdim=True)
        
        z_diff = torch.relu(max_inv_z - pixel_inv_z) 
        z_blend_weights = torch.clamp(self.ONES_CH - (z_diff * 100.0), min=0.0, max=1.0)
        z_mask = mask * z_blend_weights 

        # --- カラーの合成 (Native reduce_sum) ---
        R_low = torch.sum(final_R * z_mask, dim=1, keepdim=True)
        G_low = torch.sum(final_G * z_mask, dim=1, keepdim=True)
        B_low = torch.sum(final_B * z_mask, dim=1, keepdim=True)
        
        mask_w_low = torch.sum(z_mask, dim=1, keepdim=True)
        
    # --- 高解像度へのアップサンプリング ---
        # 1. チャンネル方向に結合 [1, 5, 256, 256]
        combined_low = torch.cat([R_low, G_low, B_low, mask_w_low, max_inv_z], dim=1)
        
        # 2. 一括でアップサンプリング [1, 5, 1024, 1024]
        combined_high = F.interpolate(
            combined_low, 
            size=(self.target_height, self.target_width), 
            mode='bilinear', 
            align_corners=False
        )
        
    # 1. まず split して変数に代入
        R, G, B, mask_w, max_inv_z_out = torch.split(combined_high, 1, dim=1)
        
        # 2. その後に return する
        return R, G, B, mask_w, max_inv_z_out