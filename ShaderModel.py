import torch
import torch.nn as nn

class ANE3DRenderer64(nn.Module):
    def __init__(self, width=256, height=256):
        super().__init__()
        self.width = width
        self.height = height
        
        # 座標バッファ (ANEのFP16ネイティブ用に[1, 1, H, W]で初期化)
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

        # 1. エッジ計算 (動的Convを廃止し、ANEが最も得意なブロードキャスト積和に)
        edges0 = a0 * self.x_coords + b0 * self.y_coords + c0
        edges1 = to_fp16(A1) * self.x_coords + to_fp16(B1) * self.y_coords + to_fp16(C1)
        edges2 = to_fp16(A2) * self.x_coords + to_fp16(B2) * self.y_coords + to_fp16(C2)

        # 2. マスク生成 (💡それぞれの乗算前に0.0〜1.0へクランプし、FP16上限突破によるNaNを根絶)
        valid_mask = torch.clamp((a0 * a0 + b0 * b0) * 100.0, min=0.0, max=1.0)
        
        mask0 = torch.clamp(edges0 * 100.0, min=0.0, max=1.0)
        mask1 = torch.clamp(edges1 * 100.0, min=0.0, max=1.0)
        mask2 = torch.clamp(edges2 * 100.0, min=0.0, max=1.0)
        
        # 0.0〜1.0 同士の安全な掛け算 (infが発生しないため、inf * 0.0 -> nan に絶対になりません)
        mask = mask0 * mask1 * mask2 * valid_mask

        # 3. 重み（重心座標）の計算 (💡絶対値 + 0.02 の足し算ガードで逆数オーバーフローを完全遮断)
        total_area = torch.abs(edges0 + edges1 + edges2) + 0.02
        inv_area = torch.reciprocal(total_area) # ANE専用の高速逆数ユニットを使用
        
        w0 = edges1 * inv_area
        w1 = edges2 * inv_area
        w2 = edges0 * inv_area

        # 4. 深度(Z)バッファ計算 (加算と乗算をインプレイス化してメモリ削減)
        pixel_inv_z = (to_fp16(p0_iz) * w0)
        pixel_inv_z.add_(to_fp16(p1_iz) * w1)
        pixel_inv_z.add_(to_fp16(p2_iz) * w2)
        pixel_inv_z.mul_(mask)

        # 5. テクスチャ座標の重心補間 ＆ サンプリング
        u_interp = (to_fp16(U0) * w0) + (to_fp16(U1) * w1) + (to_fp16(U2) * w2)
        v_interp = (to_fp16(V0) * w0) + (to_fp16(V1) * w1) + (to_fp16(V2) * w2)
        
        # ピクセルごとに補間された2次元のUV強度をテクスチャに直積
        sampled_texture = torch.clamp_(processed_texture * u_interp * v_interp, min=0.0, max=1.0)

        # 6. Z-Buffer テスト (ここだけ1x1 Convでチャンネルを[1, 1, H, W]へ集約)
        sum_inv_z = torch.conv2d(pixel_inv_z, self.sum_kernel)
        z_diff = torch.relu_(sum_inv_z - pixel_inv_z)

        z_blend_weights = torch.clamp_(1.0 - (z_diff * 10.0), min=0.0, max=1.0)
        mask.mul_(z_blend_weights) # mask テンソルを z_mask として再利用してメモリ削減

        # 7. 最終出力の集約 (R_full などの巨大な複製テンソルをすべて排除)
        color_payload = sampled_texture * mask
        
        R = torch.conv2d(color_payload, self.sum_kernel)
        G = torch.conv2d(color_payload, self.sum_kernel)
        B = torch.conv2d(color_payload, self.sum_kernel)
        mask_w = torch.conv2d(mask, self.sum_kernel)
        
        max_inv_z = torch.conv2d(pixel_inv_z * z_blend_weights, self.sum_kernel)
        
        return R, G, B, mask_w, max_inv_z
