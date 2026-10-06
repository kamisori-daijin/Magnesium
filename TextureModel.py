import torch
import torch.nn as nn

class ANETextureProcessor(nn.Module):
    def __init__(self, max_polygons=256):
        super().__init__()
        self.max_polygons = max_polygons

        self.expand_conv = nn.Conv2d(3, self.max_polygons, kernel_size=1, bias=None)

        with torch.no_grad():
            weight = torch.zeros(self.max_polygons, 3, 1, 1)
            
            for i in range(self.max_polygons):
                if i % 3 == 0:
                    weight[i, 0, 0, 0] = 1.0
                elif i % 3 == 1:
                    weight[i, 1, 0, 0] = 1.0
                else:
                    weight[i, 2, 0, 0] = 1.0
                    
            self.expand_conv.weight.copy_(weight)

    def forward(self, raw_image):
        return self.expand_conv(raw_image)