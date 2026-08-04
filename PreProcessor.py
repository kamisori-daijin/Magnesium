import torch
import torch.nn as nn
import torch.nn.functional as F

class ANE3DPreProcessor64(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_raster_faces = 64
        
    # Input: 4 channels (X, Y, Z, W)
    # Output: 4 channels (X_c, Y_c, Z_c, W_c) aligned to multiples of 4
    # By setting groups=1, the specified 4x4 weight (MVP matrix) is 
    # multiplied simultaneously and in parallel for each vertex of the 64 faces.

        self.mvp_conv = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, bias=None)
        
    def forward(self, expanded_vertices, mvp_weights, colors_r, colors_g, colors_b):
        """
        expanded_vertices: [1, 4, 3, 64] -> (4ch=XYZW, Height=3 vertices, Width=64 faces)
        mvp_weights: [4, 4, 1, 1] -> Either combine and average the
        64 faces beforehand on the Swift/Python side,
        or an MVP weight tensor with objects placed in a single common space
        
        colors_r / g / b: [1, 1, 1, 64]
        """
        

        transformed = F.conv2d(expanded_vertices, mvp_weights, bias=None) # Output: [1, 4, 3, 64]
        
        X_c = transformed[:, 0:1, :, :] # [1, 1, 3, 64]
        Y_c = transformed[:, 1:2, :, :]
        Z_c = transformed[:, 2:3, :, :]
        
    
        safe_Z = torch.clamp(torch.abs(Z_c), min=1e-5)
        screen_x = X_c / safe_Z
        screen_y = Y_c / safe_Z
        inv_Z = 1.0 / safe_Z  
        
        # Slice
        p0_x, p1_x, p2_x = screen_x[:, :, 0:1, :], screen_x[:, :, 1:2, :], screen_x[:, :, 2:3, :]
        p0_y, p1_y, p2_y = screen_y[:, :, 0:1, :], screen_y[:, :, 1:2, :], screen_y[:, :, 2:3, :]
        p0_iz = inv_Z[:, :, 0:1, :] 
        
        # Compute edge functions (A, B, C)
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
        
        return (A0, B0, C0, A1, B1, C1, A2, B2, C2, 
                R, G, B, R, G, B, R, G, B, p0_iz)
