import torch
import torch.nn as nn

class ANE3DRenderer64(nn.Module):
    def __init__(self, width=256, height=256):
        super().__init__()
        self.width = width
        self.height = height
        
        y_coords = torch.linspace(1.0, -1.0, height, dtype=torch.float16).view(1, 1, height, 1)
        x_coords = torch.linspace(-1.0, 1.0, width, dtype=torch.float16).view(1, 1, 1, width)
        
        self.register_buffer("x_coords", x_coords.expand(1, 64, height, width).contiguous())
        self.register_buffer("y_coords", y_coords.expand(1, 64, height, width).contiguous())
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

        zero_fp16 = torch.tensor(0.0, dtype=torch.float16, device=a0.device)
        one_fp16 = torch.tensor(1.0, dtype=torch.float16, device=a0.device)

        # 2. マスク生成
        abs_a0 = torch.relu(a0) + torch.relu(-a0)
        abs_b0 = torch.relu(b0) + torch.relu(-b0)
        v_mask = torch.clamp(abs_a0 + abs_b0, min=zero_fp16, max=one_fp16)
        
        mask0 = torch.clamp(torch.relu(edges0), min=zero_fp16, max=one_fp16)
        mask1 = torch.clamp(torch.relu(edges1), min=zero_fp16, max=one_fp16)
        mask2 = torch.clamp(torch.relu(edges2), min=zero_fp16, max=one_fp16)
        
        raw_mask = mask0 * mask1 * mask2 * v_mask
        mask = raw_mask * raw_mask
        mask = mask * mask
        mask = torch.clamp(torch.relu(mask), min=zero_fp16, max=one_fp16)

        # 3. 重み（重心座標）の計算
        abs_sum_edges = torch.relu(edges0 + edges1 + edges2) + torch.relu(-(edges0 + edges1 + edges2))
        total_area = abs_sum_edges + 0.02
        inv_area = 1.0 / total_area
        
        w0 = edges1 * inv_area * mask
        w1 = edges2 * inv_area * mask
        w2 = edges0 * inv_area * mask

        # 4. 深度(Z)バッファ計算 (インプレース操作 add_ を通常の + に変更)
        pixel_inv_z = (to_fp16(p0_iz) * w0) + (to_fp16(p1_iz) * w1) + (to_fp16(p2_iz) * w2)

        # 5. 頂点カラーの重心補間
        color_payload_r = (to_fp16(R0) * w0) + (to_fp16(R1) * w1) + (to_fp16(R2) * w2)
        color_payload_g = (to_fp16(G0) * w0) + (to_fp16(G1) * w1) + (to_fp16(G2) * w2)
        color_payload_b = (to_fp16(B0_col) * w0) + (to_fp16(B1_col) * w1) + (to_fp16(B2_col) * w2)

        # 6. Z-Buffer テスト
        valid_mask = torch.clamp(mask, min=zero_fp16, max=one_fp16)
        
        # 手前のピクセルを強調するために、深度に大きな値を掛ける
        z_logits = pixel_inv_z * 100.0
        
        # 描画対象外のピクセルは、Softmaxの計算から除外する
        z_logits = z_logits + (valid_mask - 1.0) * 10000.0
        
        # Softmaxで、最も手前にあるピクセルのウェイトを1に近づける
        z_blend_weights = torch.nn.functional.softmax(z_logits, dim=1)
        
        final_color_r = color_payload_r * z_blend_weights
        final_color_g = color_payload_g * z_blend_weights
        final_color_b = color_payload_b * z_blend_weights
        final_mask = mask * z_blend_weights

        # 7. 最終出力の集約
        R = torch.conv2d(final_color_r, self.sum_kernel)
        G = torch.conv2d(final_color_g, self.sum_kernel)
        B = torch.conv2d(final_color_b, self.sum_kernel)
        mask_w = torch.conv2d(final_mask, self.sum_kernel)
        
        max_inv_z = torch.conv2d(pixel_inv_z * z_blend_weights, self.sum_kernel)
        
        return R, G, B, mask_w, max_inv_z