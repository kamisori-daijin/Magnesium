import torch
import torch.nn as nn

class ANE3DRenderer64(nn.Module):
    def __init__(self, width=256, height=256):
        super().__init__()
        self.width = width
        self.height = height
        
        # 💡 【真の黒幕を粉砕】チャンネル次元（dim=1）を最初から明示的に「64」に引き伸ばして保持
        # これにより、FP16命令が裏で怪しい自動ブロードキャストを行う隙を完全に無くし、
        # 0.0の掛け算で -0.0 やビット反転（NaN）が発生するルートを100%物理的に封鎖します。
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

        # 1. エッジ計算 (同じ[1, 64, H, W]形状同士の、寸分の狂いもない純粋な並列積和)
        edges0 = a0 * self.x_coords + b0 * self.y_coords + c0
        edges1 = to_fp16(A1) * self.x_coords + to_fp16(B1) * self.y_coords + to_fp16(C1)
        edges2 = to_fp16(A2) * self.x_coords + to_fp16(B2) * self.y_coords + to_fp16(C2)

        # 💡 クランプの境界値を明示的に同じデバイス・同じ型(FP16)のテンソルとして固定
        zero_fp16 = torch.tensor(0.0, dtype=torch.float16, device=a0.device)
        one_fp16 = torch.tensor(1.0, dtype=torch.float16, device=a0.device)

        # ==================================================
        # 2. マスク生成 (ANE安全化対策：負の数によるNaNバグを完全粉砕)
        # ==================================================
        # ANEが苦手な torch.abs() を使わず、ハードウェアネイティブな relu(x) + relu(-x) で絶対値を安全に再現
        abs_a0 = torch.relu(a0) + torch.relu(-a0)
        abs_b0 = torch.relu(b0) + torch.relu(-b0)
        v_mask = torch.clamp(abs_a0 + abs_b0, min=zero_fp16, max=one_fp16)
        
        # edgesが大きな負の数の際、直接clamp(min=0)するとANEで符号反転バグ（NaN）が起きるため、
        # 一度必ず torch.relu（=ハードウェアが確実に0以下を遮断する命令）を通してから上限をclampします
        mask0 = torch.clamp(torch.relu(edges0), min=zero_fp16, max=one_fp16)
        mask1 = torch.clamp(torch.relu(edges1), min=zero_fp16, max=one_fp16)
        mask2 = torch.clamp(torch.relu(edges2), min=zero_fp16, max=one_fp16)
        
        # 0.0〜1.0 同士の純粋な幾何学乗算 (絶対に数値が爆発しません)
        raw_mask = mask0 * mask1 * mask2 * v_mask
        
        # 累乗（4乗）によって境界を急激に立たせて、シャープなラスタライズ輪郭を作ります
        mask = raw_mask * raw_mask
        mask = mask * mask
        mask = torch.clamp(torch.relu(mask), min=zero_fp16, max=one_fp16)

        # ==================================================
        # 3. 重み（重心座標）の計算 (ANE安全化対策：reciprocalのバグを回避)
        # ==================================================
        # ここも abs の代わりに relu 結合を使用
        abs_sum_edges = torch.relu(edges0 + edges1 + edges2) + torch.relu(-(edges0 + edges1 + edges2))
        total_area = abs_sum_edges + 0.02
        
        # ANEコンパイラでバグりやすい torch.reciprocal の代わりに、直球の分数除算を使用
        # ANEは 1.0 / x の形式の方がハードウェア的に安定した乗算に変換されやすいです
        inv_area = 1.0 / total_area
        
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
        sampled_texture = torch.clamp_(processed_texture * u_interp * v_interp, min=zero_fp16, max=one_fp16)

        # 6. Z-Buffer テスト (ここだけ1x1 Convでチャンネルを[1, 1, H, W]へ集約)
        sum_inv_z = torch.conv2d(pixel_inv_z, self.sum_kernel)
        z_diff = torch.relu_(sum_inv_z - pixel_inv_z)

        z_blend_weights = torch.clamp_(1.0 - (z_diff * 10.0), min=zero_fp16, max=one_fp16)
        mask.mul_(z_blend_weights) 

        # 7. 最終出力の集約
        color_payload = sampled_texture * mask
        
        R = torch.conv2d(color_payload, self.sum_kernel)
        G = torch.conv2d(color_payload, self.sum_kernel)
        B = torch.conv2d(color_payload, self.sum_kernel)
        mask_w = torch.conv2d(mask, self.sum_kernel)
        
        max_inv_z = torch.conv2d(pixel_inv_z * z_blend_weights, self.sum_kernel)
        
        return R, G, B, mask_w, max_inv_z
