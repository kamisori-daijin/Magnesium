import torch
import torch.nn as nn

class ANE3DPreProcessor64(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_raster_faces = 64
        
    def forward(self, expanded_vertices, mvp_weights, colors_r, colors_g, colors_b):
        # 1. 座標変換
        transformed = torch.matmul(mvp_weights.to(torch.float16), expanded_vertices.to(torch.float16)) # -> [1, 64, 4, 3]
        
        # 各軸の切り出し
        X_c = transformed[:, :, 0, :] # -> [1, 64, 3]
        Y_c = transformed[:, :, 1, :] # -> [1, 64, 3]
        W_c = transformed[:, :, 3, :] # -> [1, 64, 3]
        
        # 💡 【究極の修正】底上げの定数を 1e-5 ➔ 0.02 に引き上げます！
        # W_c が完全に 0.0 の未使用ポリゴンであっても、分母は確実に 0.02 になります。
        # これにより 1.0 / 0.02 = 50.0 となり、FP16の限界（65504）を絶対に突破せず、
        # 180個発生していた inf をモデルの内部で根こそぎ完全消滅させます。
        safe_W = torch.abs(W_c) + 0.02 # -> [1, 64, 3]
        
        screen_x = X_c / safe_W  # -> [1, 64, 3]
        screen_y = Y_c / safe_W  # -> [1, 64, 3]
        inv_Z = 1.0 / safe_W     # -> [1, 64, 3]
        
        # 各頂点 [1, 64, 1, 1] へのビュー変形
        p0_x = screen_x[:, :, 0].view(1, 64, 1, 1)
        p1_x = screen_x[:, :, 1].view(1, 64, 1, 1)
        p2_x = screen_x[:, :, 2].view(1, 64, 1, 1)
        
        p0_y = screen_y[:, :, 0].view(1, 64, 1, 1)
        p1_y = screen_y[:, :, 1].view(1, 64, 1, 1)
        p2_y = screen_y[:, :, 2].view(1, 64, 1, 1)
        
        p0_iz = inv_Z[:, :, 0].view(1, 64, 1, 1)
        p1_iz = inv_Z[:, :, 1].view(1, 64, 1, 1)
        p2_iz = inv_Z[:, :, 2].view(1, 64, 1, 1)

        # 2. エッジ計算
        A0 = p0_y - p1_y
        B0 = p1_x - p0_x
        C0 = -(A0 * p0_x + B0 * p0_y)
        
        A1 = p1_y - p2_y
        B1 = p2_x - p1_x
        C1 = -(A1 * p1_x + B1 * p1_y)
        
        A2 = p2_y - p0_y
        B2 = p0_x - p2_x
        C2 = -(A2 * p2_x + B2 * p2_y)
        
        R, G, B = colors_r.to(torch.float16), colors_g.to(torch.float16), colors_b.to(torch.float16)

        return (A0, B0, C0, A1, B1, C1, A2, B2, C2, 
                R, G, B, R, G, B, R, G, B, p0_iz, p1_iz, p2_iz)
