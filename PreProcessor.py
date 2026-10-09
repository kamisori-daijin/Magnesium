import torch
import torch.nn as nn

class ANE3DPreProcessor(nn.Module):
    def __init__(self, max_polygons=256):
        super().__init__()
        self.max_polygons = max_polygons
        
    def forward(self, expanded_vertices, mvp_weights, colors_r, colors_g, colors_b):
        """
        Args:
            expanded_vertices: [1, max_polygons, 4, 3] 
            mvp_weights:       [1, max_polygons, 4, 4]  
            colors_r / g / b:  [1, max_polygons, 1, 1]
        """
        P = self.max_polygons
        
        # =====================================================================
        # 1. Matrix ops (matmul): MVP行列の乗算
        # =====================================================================
        # 4次元のままだとコンパイラがバッチ処理の解釈に失敗してCPUに落とすリスクがあるため、
        # ANEが最も得意とする3次元のバッチ行列積 [P, N, M] の形状へ view (Layout) 変更。
        w_mat = mvp_weights.view(P, 4, 4)
        v_mat = expanded_vertices.view(P, 4, 3)
        
        # [P, 4, 4] x [P, 4, 3] -> [P, 4, 3]
        # ANEのMPE（行列演算コア）がネイティブ駆動し、一瞬で並列計算されます。
        transformed_raw = torch.matmul(w_mat, v_mat)
        
        # 後続の要素別演算（Elementwise ALU）に流すため、形状を [1, P, 4, 3] に戻す (Layout)
        transformed = transformed_raw.view(1, P, 4, 3)
        
        # =====================================================================
        # 2. Elementwise ALU & Layout: 座標切り出し
        # =====================================================================
        X_c = transformed[:, :, 0:1, :] # [1, P, 1, 3]
        Y_c = transformed[:, :, 1:2, :] # [1, P, 1, 3]
        Z_c = transformed[:, :, 2:3, :] # [1, P, 1, 3]
        W_c = transformed[:, :, 3:4, :] # [1, P, 1, 3]
        
        safe_W = torch.clamp(torch.abs(W_c), min=1e-5)
        
        # =====================================================================
        # 3. Activations (dedicated hardware): 逆数計算 (reciprocal)
        # =====================================================================
        inv_W = torch.reciprocal(safe_W) # ANE専用の高速ハードウェアによる逆数演算
        
        # =====================================================================
        # 4. Elementwise ALU: スクリーン座標変換
        # =====================================================================
        screen_x = X_c * inv_W
        screen_y = Y_c * inv_W
        inv_Z = inv_W     
        
        # 頂点ごとの要素分解 (Layout)
        p0_x, p1_x, p2_x = screen_x[:, :, :, 0:1], screen_x[:, :, :, 1:2], screen_x[:, :, :, 2:3]
        p0_y, p1_y, p2_y = screen_y[:, :, :, 0:1], screen_y[:, :, :, 1:2], screen_y[:, :, :, 2:3]
     
        p0_iz = inv_Z[:, :, :, 0:1] 
        p1_iz = inv_Z[:, :, :, 1:2] 
        p2_iz = inv_Z[:, :, :, 2:3] 

        # =====================================================================
        # 5. Elementwise ALU: 裏面カリング (Backface Culling)
        # =====================================================================
        edge1_x = p1_x - p0_x
        edge1_y = p1_y - p0_y
        edge2_x = p2_x - p0_x
        edge2_y = p2_y - p0_y
        
        cross_product_z = (edge1_x * edge2_y) - (edge1_y * edge2_x)
        
        # 1000倍してreluを通すことで、0未満を0、0以上を最大1にクランプ（カリングマスク生成）
        cull_mask = torch.clamp(torch.relu(cross_product_z * 1000.0), min=0.0, max=1.0)

        # =====================================================================
        # 6. Elementwise ALU: エッジ係数 (A, B, C) の計算
        # =====================================================================
        # カリングマスクをここで各係数に掛け算 (ALU: mul)
        A0 = (p0_y - p1_y) * cull_mask
        B0 = (p1_x - p0_x) * cull_mask
        C0 = -(A0 * p0_x + B0 * p0_y)
        
        A1 = (p1_y - p2_y) * cull_mask
        B1 = (p2_x - p1_x) * cull_mask
        C1 = -(A1 * p1_x + B1 * p1_y)
        
        A2 = (p2_y - p0_y) * cull_mask
        B2 = (p0_x - p2_x) * cull_mask
        C2 = -(A2 * p2_x + B2 * p2_y)
        
        # 頂点カラーは元のメモリ超節約仕様を維持、カリングマスクはレンダラー側で適用
        R, G, B = colors_r, colors_g, colors_b

        # すべて最小形状のままバラバラで return
        # メモリコピーやテンソル結合によるSRAMキャッシュの汚染を100%防ぎます
        return (A0, B0, C0, A1, B1, C1, A2, B2, C2, 
                R, G, B, R, G, B, R, G, B, p0_iz, p1_iz, p2_iz)
