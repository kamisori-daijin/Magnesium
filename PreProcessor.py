import torch
import torch.nn as nn

class ANE3DPreProcessor(nn.Module):
    def __init__(self, max_polygons=256):
        super().__init__()
        self.max_polygons = max_polygons
        
    def forward(self, expanded_vertices, mvp_weights, colors_r, colors_g, colors_b):
        """
        expanded_vertices: [1, max_polygons, 4, 3] 
        mvp_weights:       [1, max_polygons, 4, 4]  
        colors_r / g / b:  [1, max_polygons, 1, 1]
        """
        
        # No Reshape
        transformed = (
            expanded_vertices[:, :, 0:1, :] * mvp_weights[:, :, :, 0:1] +
            expanded_vertices[:, :, 1:2, :] * mvp_weights[:, :, :, 1:2] +
            expanded_vertices[:, :, 2:3, :] * mvp_weights[:, :, :, 2:3] +
            expanded_vertices[:, :, 3:4, :] * mvp_weights[:, :, :, 3:4]
        ) # Output: [1, max_polygons, 4, 3]
        
        X_c = transformed[:, :, 0:1, :]
        Y_c = transformed[:, :, 1:2, :]
        Z_c = transformed[:, :, 2:3, :]
        W_c = transformed[:, :, 3:4, :] 
        
        safe_W = torch.clamp(torch.abs(W_c), min=1e-5)
        
        # ANEの専用ハードウェア演算 (Reciprocal) を使用
        inv_W = torch.reciprocal(safe_W)
        
        screen_x = X_c * inv_W
        screen_y = Y_c * inv_W
        inv_Z = inv_W     
        
        p0_x, p1_x, p2_x = screen_x[:, :, :, 0:1], screen_x[:, :, :, 1:2], screen_x[:, :, :, 2:3]
        p0_y, p1_y, p2_y = screen_y[:, :, :, 0:1], screen_y[:, :, :, 1:2], screen_y[:, :, :, 2:3]
     
        p0_iz = inv_Z[:, :, :, 0:1] 
        p1_iz = inv_Z[:, :, :, 1:2] 
        p2_iz = inv_Z[:, :, :, 2:3] 

        # --- 裏面カリング (Backface Culling) ---
        # エッジベクトルの外積(Z成分)を計算
        edge1_x = p1_x - p0_x
        edge1_y = p1_y - p0_y
        edge2_x = p2_x - p0_x
        edge2_y = p2_y - p0_y
        
        cross_product_z = (edge1_x * edge2_y) - (edge1_y * edge2_x)
        
        # cross_product_z > 0 の場合のみ有効 (時計回り/反時計回りに応じて符号を調整してください)
        cull_mask = torch.clamp(torch.relu(cross_product_z * 1000.0), min=0.0, max=1.0)

        A0 = (p0_y - p1_y) * cull_mask
        B0 = (p1_x - p0_x) * cull_mask
        C0 = -(A0 * p0_x + B0 * p0_y)
        
        A1 = (p1_y - p2_y) * cull_mask
        B1 = (p2_x - p1_x) * cull_mask
        C1 = -(A1 * p1_x + B1 * p1_y)
        
        A2 = (p2_y - p0_y) * cull_mask
        B2 = (p0_x - p2_x) * cull_mask
        C2 = -(A2 * p2_x + B2 * p2_y)
        
        R, G, B = colors_r, colors_g, colors_b

        return (A0, B0, C0, A1, B1, C1, A2, B2, C2, 
                R, G, B, R, G, B, R, G, B, p0_iz, p1_iz, p2_iz)