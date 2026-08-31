import torch
import torch.nn as nn
import torch.nn.functional as F

class ANE3DPreProcessor64(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_raster_faces = 64
        
    def forward(self, expanded_vertices, mvp_weights, colors_r, colors_g, colors_b):
        # -----------------------------------------------------------------
        # 1. 座標変換（4次元完全維持 ＆ 3D空間の正常化）
        # -----------------------------------------------------------------
        # V: [64, 4(頂点), 4(XYZW)]
        V = expanded_vertices.squeeze(0)
        # W: [64, 4(行), 4(列)]
        W = mvp_weights.squeeze(0)
        
        # 💡 【立体化の鍵】
        # テストコード側では expanded_vertices[0, i, :3, :] = face_tensor [3, 4] としており、
        # 行（3つ目の次元）に頂点が並び、列（4つ目の次元）に XYZW が並んでいます。
        # したがって、正しい変換は「頂点ベクトル × MVP行列」つまり「V × W」です。
        transformed = torch.bmm(V, W) # 結果の形状: [64, 4(頂点), 4(XYZW)]
        
        # 💡 ANE最適化のためにバッチ次元を戻す
        transformed = transformed.unsqueeze(0) # 形状: [1, 64, 4(頂点), 4(XYZW)]
        
        # 💡 【ご指摘の修正】4次元のまま正確にスライス (H=4(頂点)、W=4(XYZW) を維持)
        # 範囲指定スライス [..., 0:1] を使うことで、次元を落とさずに [1, 64, 4, 1] を維持します。
        X_c = transformed[:, :, :, 0:1] # [1, 64, 4, 1]
        Y_c = transformed[:, :, :, 1:2] # [1, 64, 4, 1]
        W_c = transformed[:, :, :, 3:4] # [1, 64, 4, 1] (プロジェクション成分)
        
        # --- 以降は4次元テンソルのまま計算が進行 ---
        abs_W_c = torch.relu(W_c) + torch.relu(-W_c)
        safe_W = abs_W_c + 0.02
        
        # W成分での除算（遠近感の適用）
        inv_W = 1.0 / safe_W
        screen_x = X_c * inv_W
        screen_y = Y_c * inv_W
        
        # クリッピング処理（4次元を維持：[1, 64, 4, 1]）
        near_clip = 0.1
        clip_mask = torch.clamp(torch.relu(safe_W - near_clip) * 1000.0, min=0.0, max=1.0)
        
        # 💡 clip_mask_view も 4次元 [1, 64, 1, 1] を綺麗にキープ
        clip_mask_view = clip_mask[:, :, 0:1, :] 
        
        # 💡 頂点ごとの分離スライス（3つ目の次元である H次元＝頂点 をスライス）
        # 範囲指定（0:1, 1:2, 2:3）にすることで、すべて [1, 64, 1, 1] の4次元になります！
        p0_x, p1_x, p2_x = screen_x[:, :, 0:1, :], screen_x[:, :, 1:2, :], screen_x[:, :, 2:3, :]
        p0_y, p1_y, p2_y = screen_y[:, :, 0:1, :], screen_y[:, :, 1:2, :], screen_y[:, :, 2:3, :]
        p0_iz, p1_iz, p2_iz = inv_W[:, :, 0:1, :], inv_W[:, :, 1:2, :], inv_W[:, :, 2:3, :]

        # -----------------------------------------------------------------
        # 2. エッジ計算（元の直線の式を維持）
        # -----------------------------------------------------------------
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
        
        A0 = A0 * clip_mask_view
        A1 = A1 * clip_mask_view
        A2 = A2 * clip_mask_view

        R, G, B = colors_r.to(torch.float16), colors_g.to(torch.float16), colors_b.to(torch.float16)

        return (A0, B0, C0, A1, B1, C1, A2, B2, C2, 
                R, G, B, R, G, B, R, G, B, p0_iz, p1_iz, p2_iz)
