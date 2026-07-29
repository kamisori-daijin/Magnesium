import torch
import torch.nn as nn

class ANETextureProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        # 🚀 ANEが世界で一番大得意な「1x1 2D畳み込み（Conv2d）」
        # 入力: 生のRGB画像（3チャンネル） ➔ 出力: ANEアライメント（64チャンネル）
        self.expand_conv = nn.Conv2d(3, 64, kernel_size=1, bias=None)
        
        # 【低レイヤーハック】
        # 生のR, G, Bデータを、64chの並列空間へ綺麗に巡回（リピート）して配置する初期重みを固定設定
        with torch.no_grad():
            weight = torch.zeros(64, 3, 1, 1)
            for i in range(64):
                weight[i, i % 3, 0, 0] = 1.0
            self.expand_conv.weight.copy_(weight)

    def forward(self, raw_image):
        """
        raw_image: [Batch=1, Channel=3, H=256, W=256] (Swiftから届く生のRGB画像)
        """
        # ANEの物理回路を一撃で直撃し、 Instrumentsで見たあの美しい「でかい1ブロック」を生成！
        # 出力形状: [1, 64, 256, 256] ➔ 完璧な8.4MBのバッキングが確定します。
        return self.expand_conv(raw_image)
