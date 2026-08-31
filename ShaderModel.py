import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ANE3DRenderer64(nn.Module):
    def __init__(self, width=256, height=256, upscale_factor=2):
        super().__init__()
        self.width = width
        self.height = height
        self.upscale_factor = upscale_factor  # 拡大倍率（2倍なら512x512出力）
        
        # 1. 座標バッファの初期化
        y_coords = torch.linspace(1.0, -1.0, height).view(1, 1, height, 1)
        x_coords = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
        self.register_buffer("pixel_coords", torch.cat([
            x_coords.expand(1, 1, height, width),
            y_coords.expand(1, 1, height, width),
            torch.ones(1, 1, height, width)
        ], dim=1))
        
        # 2. ANEハック：NumPyで定数カーネルを完全作成（計算グラフを一切汚さない）
        out_channels = 1 * (upscale_factor ** 2)  # 2倍なら4ch
        ch_per_group = 64 // out_channels        # 16chずつまとめる
        
        # 2-a. 64chを4つのグループに分けて足し戻すための 1x1 Conv カーネル
        np_shuffle_kernel = np.zeros((out_channels, 64), dtype=np.float32)
        for i in range(out_channels):
            np_shuffle_kernel[i, i*ch_per_group : (i+1)*ch_per_group] = 1.0
        np_shuffle_kernel = np_shuffle_kernel[:, :, np.newaxis, np.newaxis]
        
        # 2-b. Zバッファ判定用：全64chを一気に1chに足し戻すための 1x1 Conv カーネル
        np_sum_kernel = np.ones((1, 64, 1, 1), dtype=np.float32)
        
        # 3. PyTorchテンソル（float16）へ一発変換してバッファ登録
        self.register_buffer("shuffle_kernel", torch.from_numpy(np_shuffle_kernel).to(torch.float16))
        self.register_buffer("sum_kernel", torch.from_numpy(np_sum_kernel).to(torch.float16))
        
        # 4. ピクセルシャッフルレイヤー
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, 
                A0, B0, C0, A1, B1, C1, A2, B2, C2, 
                R0, G0, B0_col, R1, G1, B1_col, R2, G2, B2_col,
                p0_iz, p1_iz, p2_iz,
                U0, V0, U1, V1, U2, V2,
                processed_texture):
        
        # --- [低解像度空間 (256x256) での重い演算ここから] ---
        
        # 1. Calculate Edges
        def compute_edges(A, B, C):
            weight = torch.cat([A, B, C], dim=1).permute(3, 1, 0, 2).contiguous()
            return F.conv2d(self.pixel_coords, weight, bias=None)

        edges0 = compute_edges(A0, B0, C0)
        edges1 = compute_edges(A1, B1, C1)
        edges2 = compute_edges(A2, B2, C2)

        # 2. Mask
        valid_mask = torch.clamp(torch.relu((A0**2 + B0**2) * 100.0), min=0.0, max=1.0).permute(3, 1, 0, 2)
        inside_cw = torch.relu(edges0 * 100.0) * torch.relu(edges1 * 100.0) * torch.relu(edges2 * 100.0)
        mask = torch.clamp(inside_cw * valid_mask, min=0.0, max=1.0)

        total_area = torch.clamp(edges0 + edges1 + edges2, min=1e-5)
        inv_area = torch.pow(total_area, -1.0)
        
        # Blend Weights
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

        # 6. Z Buffer ＆ 1x1 Conv（ANE専用に事前定義した全加算カーネルを使用）
        sum_inv_z = F.conv2d(pixel_inv_z, self.sum_kernel, bias=None)
        z_diff = torch.relu(sum_inv_z - pixel_inv_z)

        sharpness = 10.0 
        z_blend_weights = torch.clamp(1.0 - (z_diff * sharpness), min=0.0, max=1.0)
        z_mask = mask * z_blend_weights

        # 7. データのパッキング
        R_full = sampled_texture * z_mask
        G_full = sampled_texture * z_mask
        B_full = sampled_texture * z_mask
        mask_full = z_mask
        
        # --- [低解像度空間 (256x256) のまま 1x1 Conv でチャンネル圧縮] ---
        # ANEが最も得意とする1x1 Convで、64chを一気に4ch（PixelShuffle用）へ集約
        R_low = F.conv2d(R_full, self.shuffle_kernel, bias=None)
        G_low = F.conv2d(G_full, self.shuffle_kernel, bias=None)
        B_low = F.conv2d(B_full, self.shuffle_kernel, bias=None)
        mask_low = F.conv2d(mask_full, self.shuffle_kernel, bias=None)
        max_inv_z_low = F.conv2d(pixel_inv_z * z_blend_weights, self.shuffle_kernel, bias=None)
        
        # --- [最終出力直前：高解像度空間 (512x512) への並び替え] ---
        # 算術演算を行わず、メモリの再配置だけで4ch（256x256）を1ch（512x512）に超解像化
        R_out = self.pixel_shuffle(R_low)
        G_out = self.pixel_shuffle(G_low)
        B_out = self.pixel_shuffle(B_low)
        mask_w_out = self.pixel_shuffle(mask_low)
        max_inv_z_out = self.pixel_shuffle(max_inv_z_low)
        
        return R_out, G_out, B_out, mask_w_out, max_inv_z_out
