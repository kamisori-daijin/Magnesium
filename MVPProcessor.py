import torch
import torch.nn as nn

class ANEMVPProcessor(nn.Module):
    def __init__(self, max_vertices=65536):
        """
        max_vertices: ANE側に確保させる動的頂点インプットの最大枠。
                      標準的な1メッシュの上限である65536頂点をデフォルトに設定。
        """
        super().__init__()
        self.max_vertices = max_vertices

    def forward(self, camera_matrix, vertex_buffer):
        """
        camera_matrix: [4, 4] (MVP行列 / float16)
        vertex_buffer: [1, 4, max_vertices] (生のXYZW座標配列 / float16)
        """
        # ANEに1x1 Convとして100%美しく最適化させるため、次元を並び替えます
        # [1, 4, max_vertices] -> [max_vertices, 4]
        v_reshaped = vertex_buffer.squeeze(0).permute(1, 0)
        
        # ij (4x4カメラ行列) と vj (max_vertices × 4次元) のお尻の次元(4)を縮約
        # ANEの並列積和演算コアが、数万個の頂点を完全に同時並行で一撃MVP変換します！
        transformed = torch.einsum('ij,vj->vi', camera_matrix, v_reshaped)
        
        # 再び元のレイアウト [1, 4, max_vertices] に戻して返却
        output_buffer = transformed.permute(1, 0).unsqueeze(0)
        
        return output_buffer
