import torch
import torch.nn as nn
import torch.nn.functional as F

class ANE3DRenderer64(nn.Module):
    def __init__(self, width=256, height=256):
        super().__init__()
        self.width = width
        self.height = height
        
        # 最初からfloat16で座標マップを作成し、Castオーバーヘッドをゼロに
        y_coords = torch.linspace(1.0, -1.0, height, dtype=torch.float16).view(1, 1, height, 1)
        x_coords = torch.linspace(-1.0, 1.0, width, dtype=torch.float16).view(1, 1, 1, width)
        
        # [1, 3, height, width]
        self.register_buffer("pixel_coords", torch.cat([
            x_coords.expand(1, 1, height, width),
            y_coords.expand(1, 1, height, width),
            torch.ones(1, 1, height, width, dtype=torch.float16)
        ], dim=1))
        
        # 1x1 Convで64chの合計(Sum)を一撃で行うためのカーネル
        self.register_buffer("sum_kernel", torch.ones(1, 64, 1, 1, dtype=torch.float16))

    def forward(self, 
                weights_0, weights_1, weights_2, # 前もって [64, 3, 1, 1] に変形済みのエッジ係数
                p0_iz, p1_iz, p2_iz,
                U0, V0, U1, V1, U2, V2,
                processed_texture):
        
        # 1. Calculate Edges (forward内での重みのpermute/contiguousを完全に排除)
        # weights_X は [64, 3, 1, 1] なので、そのままF.conv2dの weight に流せる
        edges0 = F.conv2d(self.pixel_coords, weights_0, bias=None)
        edges1 = F.conv2d(self.pixel_coords, weights_1, bias=None)
        edges2 = F.conv2d(self.pixel_coords, weights_2, bias=None)

        # 2. Mask
        # 有効ポリゴン判定をtotal_areaから誘導し、A0**2などの重い計算をカット
        total_area = edges0 + edges1 + edges2
        valid_mask = torch.clamp(total_area * 100.0, min=0.0, max=1.0)
        
        # インサイド判定 (ReLUの多重掛け算)
        inside_cw = torch.relu(edges0) * torch.relu(edges1) * torch.relu(edges2)
        mask = torch.clamp(inside_cw * valid_mask * 1000.0, min=0.0, max=1.0) # [1, 64, 256, 256]

        # 3. Blend Weights
        # torch.powを排除し、シンプルな除算に変更
        inv_area = 1.0 / torch.clamp(total_area, min=1e-5)
        w0 = edges1 * inv_area
        w1 = edges2 * inv_area
        w2 = edges0 * inv_area

        # 4. Z-Buffer
        p0_z_space = p0_iz.view(1, 64, 1, 1) * w0
        p1_z_space = p1_iz.view(1, 64, 1, 1) * w1
        p2_z_space = p2_iz.view(1, 64, 1, 1) * w2
        pixel_inv_z = (p0_z_space + p1_z_space + p2_z_space) * mask

        # 5. UV Gradients & Sampling (RGBを分離せずまとめて処理可能)
        u_gradient = (U0.view(1, 64, 1, 1) * w0 + U1.view(1, 64, 1, 1) * w1 + U2.view(1, 64, 1, 1) * w2)
        v_gradient = (V0.view(1, 64, 1, 1) * w0 + V1.view(1, 64, 1, 1) * w1 + V2.view(1, 64, 1, 1) * w2)
        
        # カラーテクスチャサンプリング（仮）
        sampled_texture = torch.clamp((processed_texture * u_gradient + processed_texture * v_gradient) * 0.5, min=0.0, max=1.0)

        # 6. Z-Test & Blend Weights
        sum_inv_z = F.conv2d(pixel_inv_z, self.sum_kernel, bias=None)
        z_diff = torch.relu(sum_inv_z - pixel_inv_z)

        sharpness = 10.0 
        z_blend_weights = torch.clamp(1.0 - (z_diff * sharpness), min=0.0, max=1.0)
        z_mask = mask * z_blend_weights

        # 7. 最終リダクション (RGBの重複計算をまとめ、メモリ確保を減らす)
        # texture * z_mask を一発で計算
        color_masked = sampled_texture * z_mask # [1, 64, 256, 256]
        
        # 1x1 ConvによるSumリダクション
        R = F.conv2d(color_masked, self.sum_kernel, bias=None)
        G = F.conv2d(color_masked, self.sum_kernel, bias=None) # ※テクスチャのチャンネル構造に応じて調整
        B = F.conv2d(color_masked, self.sum_kernel, bias=None)
        mask_w = F.conv2d(z_mask, self.sum_kernel, bias=None)
        
        max_inv_z = F.conv2d(pixel_inv_z * z_blend_weights, self.sum_kernel, bias=None)
        
        return R, G, B, mask_w, max_inv_z
