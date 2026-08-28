import torch
import torch.nn as nn

class ANE3DRenderer64(nn.Module):
    def __init__(self, width=256, height=256):
        super().__init__()
        self.width = width
        self.height = height
        
        # 座標バッファ
        y_coords = torch.linspace(1.0, -1.0, height, dtype=torch.float16).view(1, 1, height, 1)
        x_coords = torch.linspace(-1.0, 1.0, width, dtype=torch.float16).view(1, 1, 1, width)
        
        self.register_buffer("x_coords", x_coords)
        self.register_buffer("y_coords", y_coords)
        self.register_buffer("sum_kernel", torch.ones(1, 64, 1, 1, dtype=torch.float16))

    def forward(self, 
                A0, B0, C0, A1, B1, C1, A2, B2, C2, 
                R0, G0, B0_col, R1, G1, B1_col, R2, G2, B2_col,
                p0_iz, p1_iz, p2_iz,
                U0, V0, U1, V1, U2, V2,
                processed_texture):
        
        def to_fp16(t):
            return t.to(torch.float16)

        a0, b0, c0 = to_fp16(A0), to_fp16(B0), to_fp16(C0)

        # 1. エッジ計算
        edges0 = a0 * self.x_coords + b0 * self.y_coords + c0
        edges1 = to_fp16(A1) * self.x_coords + to_fp16(B1) * self.y_coords + to_fp16(C1)
        edges2 = to_fp16(A2) * self.x_coords + to_fp16(B2) * self.y_coords + to_fp16(C2)

        # 💡 【改善点】**2 を排除し、a0 * a0 に変更して pow を抹殺
        valid_mask = torch.clamp_((a0 * a0 + b0 * b0) * 100.0, min=0.0, max=1.0)
        
        # 2. マスク生成
        mask = torch.relu(edges0 * 100.0)
        mask.mul_(torch.relu(edges1 * 100.0))
        mask.mul_(torch.relu(edges2 * 100.0))
        mask.mul_(valid_mask)
        mask = torch.clamp_(mask, min=0.0, max=1.0)

        # 3. 重み（重心座標）の計算
        total_area = torch.clamp(edges0 + edges1 + edges2, min=1e-5)
        inv_area = torch.reciprocal(total_area)
        
        w0 = edges1 * inv_area
        w1 = edges2 * inv_area
        w2 = edges0 * inv_area

        # 4. 深度(Z)バッファ計算
        pixel_inv_z = (to_fp16(p0_iz) * w0)
        pixel_inv_z.add_(to_fp16(p1_iz) * w1)
        pixel_inv_z.add_(to_fp16(p2_iz) * w2)
        pixel_inv_z.mul_(mask)

        # 5. テクスチャ座標のグラジエント
        u_gradient = (to_fp16(U0) * w0) + (to_fp16(U1) * w1) + (to_fp16(U2) * w2)
        v_gradient = (to_fp16(V0) * w0) + (to_fp16(V1) * w1) + (to_fp16(V2) * w2)
        
        sampled_texture = torch.clamp_((u_gradient + v_gradient) * (processed_texture * 0.5), min=0.0, max=1.0)

        # 🎨 【将来用】頂点カラーの線形補間（ブレンド準備）
        # 必要になったら下の3行のコメントアウトを解除してください
        # r_blend = to_fp16(R0) * w0 + to_fp16(R1) * w1 + to_fp16(R2) * w2
        # g_blend = to_fp16(G0) * w0 + to_fp16(G1) * w1 + to_fp16(G2) * w2
        # b_blend = to_fp16(B0_col) * w0 + to_fp16(B1) * w1 + to_fp16(B2_col) * w2
        # 例: sampled_texture = sampled_texture * r_blend (乗算ブレンドの場合)

        # 6. Z-Buffer テスト
        sum_inv_z = torch.conv2d(pixel_inv_z, self.sum_kernel)
        z_diff = torch.relu_(sum_inv_z - pixel_inv_z)

        z_blend_weights = torch.clamp_(1.0 - (z_diff * 10.0), min=0.0, max=1.0)
        mask.mul_(z_blend_weights)

        # 7. 最終出力の集約
        color_payload = sampled_texture * mask
        
        R = torch.conv2d(color_payload, self.sum_kernel)
        G = torch.conv2d(color_payload, self.sum_kernel)
        B = torch.conv2d(color_payload, self.sum_kernel)
        mask_w = torch.conv2d(mask, self.sum_kernel)
        
        max_inv_z = torch.conv2d(pixel_inv_z * z_blend_weights, self.sum_kernel)
        
        return R, G, B, mask_w, max_inv_z
