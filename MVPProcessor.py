import torch
import torch.nn as nn
import torch.nn.functional as F

class ANEMVPProcessor(nn.Module):
    def __init__(self, max_vertices=65536):
        super().__init__()
        self.max_vertices = max_vertices

    def forward(self, camera_matrix, vertex_buffer):
        """
        camera_matrix: MVP matrix
        vertex_buffer: [1, 4, 1, max_vertices]
        """
        # Reshape camera_matrix [4, 4] to 1x1 Conv weight shape [4, 4, 1, 1]
        camera_weight = camera_matrix.view(4, 4, 1, 1)
        
        # Use F.conv2d for ANE optimization
        transformed = F.conv2d(vertex_buffer, camera_weight, bias=None) # Output: [1, 4, 1, max_vertices]
        
        X_c = transformed[:, 0:1, :, :]
        Y_c = transformed[:, 1:2, :, :]
        Z_c = transformed[:, 2:3, :, :]

        # Safety measure for division by zero and back-face clipping
        safe_Z = torch.clamp(torch.abs(Z_c), min=1e-5)
        
        screen_x = X_c / safe_Z
        screen_y = Y_c / safe_Z
        
        # Concatenate and return as [1, 3, 1, max_vertices]
        output_buffer = torch.cat([screen_x, screen_y, Z_c], dim=1)
        
        return output_buffer