import torch
import torch.nn as nn

class ANEMVPProcessor(nn.Module):
    def __init__(self, max_vertices=65536):
        super().__init__()
        self.max_vertices = max_vertices

    def forward(self, camera_matrix, vertex_buffer):
        """
        camera_matrix: (MVP行列)
        vertex_buffer: [1, 4, max_vertices] (生のXYZW座標配列)
        """
        # 1. ANEの全結合（1x1 Conv互換）で一斉に行列乗算
        v_reshaped = vertex_buffer.squeeze(0).permute(1, 0)
        transformed = torch.einsum('ij,vj->vi', camera_matrix, v_reshaped)
        transformed = transformed.permute(1, 0).unsqueeze(0) # [1, 4, max_vertices]
        
        X_c = transformed[:, 0:1, :]
        Y_c = transformed[:, 1:2, :]
        Z_c = transformed[:, 2:3, :]

        # 2. ★【超ストレート仕様】符号反転を一切やめ、純粋にZの距離を奥行きにする
        # Z_c が 0 に近づいたときのゼロ除算を防ぐ安全対策（ANEセーフティ）
        safe_Z = torch.clamp(Z_c, min=1e-5)
        
        # ANEが大好きな要素ごとのディヴィジョン（割り算）
        screen_x = X_c / safe_Z
        screen_y = Y_c / safe_Z
        
        # [1, 3, max_vertices] の形状（X, Y, 深度Z）にまとめて返却
        output_buffer = torch.cat([screen_x, screen_y, Z_c], dim=1)
        
        return output_buffer
