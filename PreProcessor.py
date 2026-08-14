import torch
import torch.nn as nn
import torch.nn.functional as F

class ANE3DPreProcessor64(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_raster_faces = 64
        
    def forward(self, expanded_vertices, mvp_weights, colors_r, colors_g, colors_b):
        """
        expanded_vertices: [1, 4, 3, 64] 
        mvp_weights:       [1, 4, 4, 1, 64] 
        colors_r / g / b:  [1, 1, 1, 64]
        """
     
        # vertex [1, 4, 3, 64] ➔ [1, 1, 4, 3, 64] 
        v_expanded = expanded_vertices.unsqueeze(1)
  
        transformed = torch.sum(mvp_weights * v_expanded, dim=2) # Output: [1, 4, 3, 64]
        
        X_c = transformed[:, 0:1, :, :] # [1, 1, 3, 64]
        Y_c = transformed[:, 1:2, :, :]
        Z_c = transformed[:, 2:3, :, :]
        W_c = transformed[:, 3:4, :, :] 
        
        safe_W = torch.clamp(torch.abs(W_c), min=1e-5)
        screen_x = X_c / safe_W 
        screen_y = Y_c / safe_W
        inv_Z = 1.0 / safe_W     # z_weights
        
        p0_x, p1_x, p2_x = screen_x[:, :, 0:1, :], screen_x[:, :, 1:2, :], screen_x[:, :, 2:3, :]
        p0_y, p1_y, p2_y = screen_y[:, :, 0:1, :], screen_y[:, :, 1:2, :], screen_y[:, :, 2:3, :]
     
        p0_iz = inv_Z[:, :, 0:1, :] 
        p1_iz = inv_Z[:, :, 1:2, :] 
        p2_iz = inv_Z[:, :, 2:3, :] 

        A0 = p0_y - p1_y
        B0 = p1_x - p0_x
        C0 = -(A0 * p0_x + B0 * p0_y)
        
        A1 = p1_y - p2_y
        B1 = p2_x - p1_x
        C1 = -(A1 * p1_x + B1 * p1_y)
        
        A2 = p2_y - p0_y
        B2 = p0_x - p2_x
        C2 = -(A2 * p2_x + B2 * p2_y)
        
        R, G, B = colors_r, colors_g, colors_b

        # Bounding Box Calculation
        x_coords = torch.cat([p0_x, p1_x, p2_x], dim=2) # [1, 1, 3, 64]
        y_coords = torch.cat([p0_y, p1_y, p2_y], dim=2) # [1, 1, 3, 64]
        
        min_x = torch.min(x_coords, dim=2, keepdim=True)[0] # [1, 1, 1, 64]
        max_x = torch.max(x_coords, dim=2, keepdim=True)[0]
        min_y = torch.min(y_coords, dim=2, keepdim=True)[0]
        max_y = torch.max(y_coords, dim=2, keepdim=True)[0]
        # ---------------------------------------

        return (A0, B0, C0, A1, B1, C1, A2, B2, C2, 
                R, G, B, R, G, B, R, G, B, p0_iz, p1_iz, p2_iz,
                min_x, max_x, min_y, max_y) 