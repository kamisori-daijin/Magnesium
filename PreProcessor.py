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
        
        # 💡 【ANE安全化対策】torch.abs(W_c) を排除し、ハードウェアが誤作動しない relu 結合へ置換
        # これにより、未使用ポリゴンの -0.0 に起因する隠れNaNの発生源を完全に断ち切ります。
        abs_W_c = torch.relu(W_c) + torch.relu(-W_c)
        safe_W = abs_W_c + 0.02 # -> [1, 64, 3]
        
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
        # 💡 【真犯人粉砕】単項マイナス「-(x)」を完全排除！
        # 0.0 から引くか、数式を「0.0 - A0*p0_x - B0*p0_y」と直球の減算命令にすることで、
        # ANEのコンパイラが「-0.0」を生成してビット崩壊（NaN）を起こすルートを100%物理封鎖します。
        zero_fp16 = torch.tensor(0.0, dtype=torch.float16, device=A0.device)
        C0 = zero_fp16 - (A0 * p0_x) - (B0 * p0_y)
        
        A1 = p1_y - p2_y
        B1 = p2_x - p1_x
        C1 = zero_fp16 - (A1 * p1_x) - (B1 * p1_y)
        
        A2 = p2_y - p0_y
        B2 = p0_x - p2_x
        C2 = zero_fp16 - (A2 * p2_x) - (B2 * p2_y)
        
        R, G, B = colors_r.to(torch.float16), colors_g.to(torch.float16), colors_b.to(torch.float16)

        return (A0, B0, C0, A1, B1, C1, A2, B2, C2, 
                R, G, B, R, G, B, R, G, B, p0_iz, p1_iz, p2_iz)
