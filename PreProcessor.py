import torch
import torch.nn as nn
import torch.nn.functional as F

class ANE3DPreProcessor64(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_raster_faces = 64
        
    def forward(self, expanded_vertices, mvp_weights, colors_r, colors_g, colors_b):
        # 1. 座標変換
        V = expanded_vertices.squeeze(0)
        W = mvp_weights.squeeze(0)
        
        # バッチ行列積 (bmm) を使用
        # これにより、64面それぞれに対して 4x4 の行列積が独立して計算されます
        transformed = torch.bmm(W, V)
        
        # 元の形 [1, 64, 4, 4] に戻す
        transformed = transformed.unsqueeze(0)
        
        # --- 以降は元のコードと同じ ---
        X_c = transformed[:, :, :, 0].unsqueeze(-1)
        Y_c = transformed[:, :, :, 1].unsqueeze(-1)
        W_c = transformed[:, :, :, 3].unsqueeze(-1)
        
        abs_W_c = torch.relu(W_c) + torch.relu(-W_c)
        safe_W = abs_W_c + 0.02
        
        # 除算を1回にまとめ、以降は乗算を使用
        inv_W = 1.0 / safe_W
        screen_x = X_c * inv_W
        screen_y = Y_c * inv_W
        
        # 💡 クリッピング処理（4次元を維持）
        near_clip = 0.1
        clip_mask = torch.clamp(torch.relu(safe_W - near_clip) * 1000.0, min=0.0, max=1.0)
        clip_mask_view = clip_mask[:, :, 0:1, :] # [1, 64, 1, 1]
        
        # 💡 スライス時に次元が落ちないように範囲指定
        p0_x, p1_x, p2_x = screen_x[:, :, 0:1, :], screen_x[:, :, 1:2, :], screen_x[:, :, 2:3, :]
        p0_y, p1_y, p2_y = screen_y[:, :, 0:1, :], screen_y[:, :, 1:2, :], screen_y[:, :, 2:3, :]
        p0_iz, p1_iz, p2_iz = inv_W[:, :, 0:1, :], inv_W[:, :, 1:2, :], inv_W[:, :, 2:3, :]

        # 2. エッジ計算
       
        A0 = p0_y - p1_y
        B0 = p1_x - p0_x
        zero_fp16 = torch.tensor(0.0, dtype=torch.float16, device=A0.device)
        C0 = zero_fp16 - (A0 * p0_x) - (B0 * p0_y)
        
        A1 = p1_y - p2_y
        B1 = p2_x - p1_x
        C1 = zero_fp16 - (A1 * p1_x) - (B1 * p1_y)
        
        A2 = p2_y - p0_y
        B2 = p0_x - p2_x
        C2 = zero_fp16 - (A2 * p2_x) - (B2 * p2_y)
        
        # クリッピングマスクを適用
        A0 = A0 * clip_mask_view
        A1 = A1 * clip_mask_view
        A2 = A2 * clip_mask_view

        R, G, B = colors_r.to(torch.float16), colors_g.to(torch.float16), colors_b.to(torch.float16)

        return (A0, B0, C0, A1, B1, C1, A2, B2, C2, 
                R, G, B, R, G, B, R, G, B, p0_iz, p1_iz, p2_iz)