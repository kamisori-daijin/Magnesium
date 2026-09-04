import torch
import torch.nn as nn

class ANETextureProcessor(nn.Module):
    def __init__(self):
        super().__init__()

        self.expand_conv = nn.Conv2d(3, 64, kernel_size=1, bias=None)

        with torch.no_grad():
            weight = torch.zeros(64, 3, 1, 1)
            
            # R, G, B をそれぞれ連続したブロックに配置する
            # これにより、レンダラー側の R_blend, G_blend, B_blend との対応が正確になります
            for i in range(64):
                if i % 3 == 0:
                    weight[i, 0, 0, 0] = 1.0  # Red
                elif i % 3 == 1:
                    weight[i, 1, 0, 0] = 1.0  # Green
                else:
                    weight[i, 2, 0, 0] = 1.0  # Blue
                    
            self.expand_conv.weight.copy_(weight)

    def forward(self, raw_image):
        """
        raw_image: [Batch=1, Channel=3, H=256, W=256] 
        """
        # [1, 64, 256, 256] 
        return self.expand_conv(raw_image)
