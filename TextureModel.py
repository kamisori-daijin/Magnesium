import torch
import torch.nn as nn
import torch.nn.functional as F

class ANETextureProcessor(nn.Module):
    def __init__(self):
        super().__init__()
      
        self.expand_conv = nn.Conv2d(3, 64, kernel_size=1, bias=None)

        with torch.no_grad():
            weight = torch.zeros(64, 3, 1, 1)
            # 最初の3チャンネルにR, G, Bをそれぞれ割り当てる
            weight[0, 0, 0, 0] = 1.0 # R
            weight[1, 1, 0, 0] = 1.0 # G
            weight[2, 2, 0, 0] = 1.0 # B
            # 4チャンネル目以降は0のまま（または必要に応じてパディング）
            self.expand_conv.weight.copy_(weight)

    def forward(self, raw_image):
        """
        raw_image: DOOM [Batch=1, Channel=3, H=400, W=640] 
        """

        square_image = F.interpolate(raw_image, size=(256, 256), mode='bilinear', align_corners=False)
        
        return self.expand_conv(square_image)