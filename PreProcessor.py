import torch
import torch.nn as nn

class ANE3DPreProcessor64(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_raster_faces = 64
        
    def forward(self, expanded_vertices, mvp_weights, colors_r, colors_g, colors_b):
        """
        💡 入力の段階で 64ch を2番目の次元（dim=1）に配置しておく！
        expanded_vertices: [1, 64, 4, 3]  (バッチ, ポリゴン, 同次座標4, 頂点3)
        mvp_weights:       [1, 64, 4, 4]  (バッチ, ポリゴン, 行列4, 列4)
        colors_r / g / b:  [1, 64, 1, 1]  (最初から完璧な形状)
        """
        
        # 1. 座標変換 (行列パラメータと頂点パラメータが同じ [1, 64, ...] なので、そのまま直積して足すだけ)
        # mvp_weights: [1, 64, 4, 4] / expanded_vertices: [1, 64, 4, 3]
        # ANEが得意な「2D平面上での要素ごとの積和」に勝手にコンパイルされます
        # ここで unsqueeze も sum(dim=2) も完全に消滅します！
        
        # 簡易的な行列乗算の代わり（ANE向けにブロードキャスト積和を行う場合）
        # ※ もし元の計算をそのまま維持するなら、次元順序を揃えた状態でmatmulするか、アインシュタイン和を使用
        # ここでは順序を揃えた状態での透過的な計算を行います
        transformed = torch.matmul(mvp_weights.to(torch.float16), expanded_vertices.to(torch.float16)) # -> [1, 64, 4, 3]
        
        # スライスするだけで、自動的に [1, 64, 1, 3] の美しい形状になります
        X_c = transformed[:, :, 0:1, :] 
        Y_c = transformed[:, :, 1:2, :] 
        W_c = transformed[:, :, 3:4, :] 
        
        safe_W = torch.clamp(torch.abs(W_c), min=1e-5)
        screen_x = X_c / safe_W  # -> [1, 64, 1, 3]
        screen_y = Y_c / safe_W  # -> [1, 64, 1, 3]
        inv_Z = 1.0 / safe_W     # -> [1, 64, 1, 3]
        
        # 3つの頂点にバラバラに切り出す (自動的に [1, 64, 1, 1] になる！)
        # view変換は一切不要です
        p0_x, p1_x, p2_x = screen_x[:, :, :, 0:1], screen_x[:, :, :, 1:2], screen_x[:, :, :, 2:3]
        p0_y, p1_y, p2_y = screen_y[:, :, :, 0:1], screen_y[:, :, :, 1:2], screen_y[:, :, :, 2:3]
        
        p0_iz, p1_iz, p2_iz = inv_Z[:, :, :, 0:1], inv_Z[:, :, :, 1:2], inv_Z[:, :, :, 2:3]

        # 2. エッジ計算 (一切のブレイクなし、完璧にスムーズな1層の並列演算)
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
