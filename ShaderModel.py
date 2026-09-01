import torch
import torch.nn as nn
import torch.nn.functional as F

class ANE3DRenderer64(nn.Module):
    def __init__(self, width=256, height=256):
        super().__init__()
        self.width = width
        self.height = height
        
        # 1. 座標系バッファ（形状固定）
        y_coords = torch.linspace(1.0, -1.0, height).view(1, 1, height, 1)
        x_coords = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
        self.register_buffer("pixel_coords", torch.cat([
            x_coords.expand(1, 1, height, width),
            y_coords.expand(1, 1, height, width),
            torch.ones(1, 1, height, width)
        ], dim=1))
        
        # 集約用カーネル
        self.register_buffer("sum_kernel", torch.ones(1, 64, 1, 1, dtype=torch.float16))
        
        # 定数スカラーの排除用バッファ（ブロードキャスト完全抑止）
        self.register_buffer("ONES_64CH", torch.ones(1, 64, height, width, dtype=torch.float16))
        self.register_buffer("SHARPNESS_SCALE", torch.full((1, 64, height, width), 10.0, dtype=torch.float16))

    def forward(self, 
                A0, B0, C0, A1, B1, C1, A2, B2, C2, 
                R0, G0, B0_col, R1, G1, B1_col, R2, G2, B2_col,
                p0_iz, p1_iz, p2_iz,
                U0, V0, U1, V1, U2, V2,
                processed_texture): # Shape: (1, 64, 256, 256) を想定
        
        # ----------------------------------------------------------------
        # 1. エッジ計算 (1x1 Conv)
        # ----------------------------------------------------------------
        def compute_edges(A, B, C):
            weight = torch.cat([A, B, C], dim=1).permute(3, 1, 0, 2).contiguous()
            return F.conv2d(self.pixel_coords, weight, bias=None)

        edges0 = compute_edges(A0, B0, C0) 
        edges1 = compute_edges(A1, B1, C1) 
        edges2 = compute_edges(A2, B2, C2) 

        # ----------------------------------------------------------------
        # 2. マスク計算 (定数倍とreluのみ。sign不要のANE最高速ルート)
        # ----------------------------------------------------------------
        valid_mask = torch.clamp(torch.relu((A0**2 + B0**2) * 100.0), min=0.0, max=1.0).permute(3, 1, 0, 2)
        inside_cw = torch.relu(edges0 * 100.0) * torch.relu(edges1 * 100.0) * torch.relu(edges2 * 100.0)
        mask = torch.clamp(inside_cw, min=0.0, max=1.0) * valid_mask 

        # ----------------------------------------------------------------
        # 3. 重心座標の計算 (powを排除、通常の順引き除算)
        # ----------------------------------------------------------------
        total_area = torch.clamp(edges0 + edges1 + edges2, min=1e-5)
        w0 = edges1 / total_area
        w1 = edges2 / total_area
        w2 = edges0 / total_area

        # ----------------------------------------------------------------
        # 4. Z Buffer (逆デプス)
        # ----------------------------------------------------------------
        p0_z_space = p0_iz.view(1, 64, 1, 1) * w0
        p1_z_space = p1_iz.view(1, 64, 1, 1) * w1
        p2_z_space = p2_iz.view(1, 64, 1, 1) * w2
        pixel_inv_z = (p0_z_space + p1_z_space + p2_z_space) * mask 

        # ----------------------------------------------------------------
        # 5. メモリコピー、軸入れ替え一切ナシの純粋積算
        # ----------------------------------------------------------------
        u_gradient = (U0.view(1, 64, 1, 1) * w0 + U1.view(1, 64, 1, 1) * w1 + U2.view(1, 64, 1, 1) * w2)
        v_gradient = (V0.view(1, 64, 1, 1) * w0 + V1.view(1, 64, 1, 1) * w1 + V2.view(1, 64, 1, 1) * w2)
        
        u_sampler = processed_texture * u_gradient
        v_sampler = processed_texture * v_gradient
        sampled_texture = torch.clamp((u_sampler + v_sampler) * 0.5, min=0.0, max=1.0)

        # ----------------------------------------------------------------
        # 6. Z Buffer ＆ ブレンドウェイト
        # ----------------------------------------------------------------
        sum_inv_z = F.conv2d(pixel_inv_z, self.sum_kernel, bias=None) 
        z_diff = torch.relu(sum_inv_z - pixel_inv_z) 

        # 動的スカラーのブロードキャストを排除し、定数バッファで演算
        z_blend_weights = torch.clamp(self.ONES_64CH - (z_diff * self.SHARPNESS_SCALE), min=0.0, max=1.0)
        z_mask = mask * z_blend_weights 

        # ----------------------------------------------------------------
        # 7. 最終画面へ1x1 ConvでR, G, Bをそれぞれ集約
        # ----------------------------------------------------------------
        R_full = sampled_texture * z_mask
        G_full = sampled_texture * z_mask
        B_full = sampled_texture * z_mask
        
        R = F.conv2d(R_full, self.sum_kernel, bias=None)
        G = F.conv2d(G_full, self.sum_kernel, bias=None)
        B = F.conv2d(B_full, self.sum_kernel, bias=None)
        mask_w = F.conv2d(z_mask, self.sum_kernel, bias=None)
        
        max_inv_z = F.conv2d(pixel_inv_z * z_blend_weights, self.sum_kernel, bias=None)
        
        return R, G, B, mask_w, max_inv_z
