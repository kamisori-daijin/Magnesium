import torch
import torch.nn as nn

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
        raw_image: [Batch=1, Channel=3, H=256, W=256] 
        """
        # [1, 64, 256, 256] 
        return self.expand_conv(raw_image)
