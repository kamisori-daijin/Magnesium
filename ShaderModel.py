import torch
import torch.nn as nn
import torch.nn.functional as f

class ANEShaderTriangle(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: Edge detection uses f.conv2d within the forward method.
        self.relu_outside = nn.ReLU()
        
        # 3rd Layer: Color Mapper (maintained as a fixed circuit with 3-channel input and 4-channel output)
        self.color_mapper = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=1, bias=True)
        self.final_clamp = nn.ReLU()
        
        self._initialize_color_weights()

    def _initialize_color_weights(self):
        with torch.no_grad():
            penalty = -5.0  
            w = torch.tensor([
                [penalty, penalty, penalty], # R (Minus if even one is outside)
                [0.0,     0.0,     0.0    ], # G
                [0.0,     0.0,     0.0    ], # B
                [penalty, penalty, penalty] # A (Minus if even one is incorrect)
            ], dtype=torch.float16).view(4, 3, 1, 1)
            
            b = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float16)
            
            self.color_mapper.weight.copy_(w)
            self.color_mapper.bias.copy_(b)

    def forward(self, x, tri_weight, tri_bias):
        # Feed in x (1, 32, 1024, 1024) and tri_weight (3, 32, 1, 1).
        edges = f.conv2d(x, tri_weight, tri_bias)
        outside_amt = self.relu_outside(edges)
        raw_rgba = self.color_mapper(outside_amt)
        return self.final_clamp(raw_rgba)
