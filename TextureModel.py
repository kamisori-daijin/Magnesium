import torch
import torch.nn as nn
import torch.nn.functional as F 

class ANETextureProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand_conv = nn.Conv2d(3, 64, kernel_size=1, bias=None)

        with torch.no_grad():
            weight = torch.zeros(64, 3, 1, 1)
            for i in range(64):
                weight[i, i % 3, 0, 0] = 1.0
            self.expand_conv.weight.copy_(weight)

    def forward(self, raw_image):
        """
        raw_image: DOOM　Size [Batch=1, Channel=3, H=200, W=320]
        """
        # 1. 1x1 Conv[1, 64, 200, 320]
        x = self.expand_conv(raw_image)
        
        
        return F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
