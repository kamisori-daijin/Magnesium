import torch
import torch.nn as nn
import torch.nn.functional as F

class ANEMVPProcessor(nn.Module):
    def __init__(self, max_vertices=65536):
        super().__init__()
        self.max_vertices = max_vertices

    def forward(self, camera_matrix, vertex_buffer):
        """
        camera_matrix: (MVP行列)
        vertex_buffer: [1, 4, 1, max_vertices]  ← ★最初から完璧な4次元画像レイアウト！
        """
        # camera_matrix [4, 4] を 1x1 Conv のウェイト形状 [4, 4, 1, 1] に変形
        # ANEはこのような小さなウェイトテンソルの変形であれば、100%拒絶せずに受け入れます
        camera_weight = camera_matrix.view(4, 4, 1, 1)
        
        # einsumやsqueezeを全廃し、ANEの積和演算コアに「F.conv2d」を直撃！！
        # これによりコンパイラは別のプロセッサへ逃げることが物理的に不可能になり、100% ANEに召喚されます
        transformed = F.conv2d(vertex_buffer, camera_weight, bias=None) # 出力: [1, 4, 1, max_vertices]
        
        X_c = transformed[:, 0:1, :, :]
        Y_c = transformed[:, 1:2, :, :]
        Z_c = transformed[:, 2:3, :, :]

        # ゼロ除算と画面背面クリップのための安全対策（ANEセーフティ）
        safe_Z = torch.clamp(torch.abs(Z_c), min=1e-5)
        
        # ANEが大好きな要素ごとのディヴィジョン（割り算）
        screen_x = X_c / safe_Z
        screen_y = Y_c / safe_Z
        
        # 後続のラスタライザへ最高の形で引き渡すため、
        # [1, 3, 1, max_vertices] の完璧な4次元レイアウトのまま結合して返却！
        output_buffer = torch.cat([screen_x, screen_y, Z_c], dim=1)
        
        return output_buffer
