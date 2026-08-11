import torch
import torch.nn as nn
import torch.nn.functional as F

class ANETextureProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand_conv = nn.Conv2d(3, 64, kernel_size=1, bias=None)

        with torch.no_grad():
            weight = torch.zeros(64, 3, 1, 1)
            
            # 0〜21: 赤 (R)
            for i in range(0, 22):
                weight[i, 0, 0, 0] = 1.0
            # 22〜43: 緑 (G)
            for i in range(22, 44):
                weight[i, 1, 0, 0] = 1.0
            # 44〜63: 青 (B)
            for i in range(44, 64):
                weight[i, 2, 0, 0] = 1.0
                
            self.expand_conv.weight.copy_(weight)

    def forward(self, raw_image):
        square_image = F.interpolate(raw_image, size=(256, 256), mode='bilinear', align_corners=False)
        return self.expand_conv(square_image)