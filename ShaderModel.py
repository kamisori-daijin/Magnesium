import torch
import torch.nn as nn
import torch.nn.functional as F

class ANE3DRenderer(nn.Module):
    def __init__(self, max_triangles=21845, width=256, height=256):
        super().__init__()
        self.max_triangles = max_triangles
        self.width = width
        self.height = height
        
        y_coords = torch.linspace(1.0, -1.0, height).view(1, 1, height, 1)
        x_coords = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
        
        X = x_coords.expand(1, 1, height, width)
        Y = y_coords.expand(1, 1, height, width)
        One = torch.ones_like(X)
        
        self.register_buffer("pixel_coords", torch.cat([X, Y, One], dim=1))

    def forward(self, transformed_vertices):
        valid_len = self.max_triangles * 3
        v_buffer = transformed_vertices[:, :, :, :valid_len]
        
        p0_X = v_buffer[:, 0:1, :, 0::3]; p0_Y = v_buffer[:, 1:2, :, 0::3]
        p1_X = v_buffer[:, 0:1, :, 1::3]; p1_Y = v_buffer[:, 1:2, :, 1::3]
        p2_X = v_buffer[:, 0:1, :, 2::3]; p2_Y = v_buffer[:, 1:2, :, 2::3]

        A0 = p0_Y - p1_Y; B0 = p1_X - p0_X; C0 = -(A0 * p0_X + B0 * p0_Y)
        A1 = p1_Y - p2_Y; B1 = p2_X - p1_X; C1 = -(A1 * p1_X + B1 * p1_Y)
        A2 = p2_Y - p0_Y; B2 = p0_X - p2_X; C2 = -(A2 * p0_X + B2 * p0_Y)

        z0 = v_buffer[:, 2:3, :, 0::3]; z1 = v_buffer[:, 2:3, :, 1::3]; z2 = v_buffer[:, 2:3, :, 2::3]
        avg_z = (z0 + z1 + z2) / 3.0
        poly_normal_Z = torch.abs(A0 * B2 - B0 * A2)

        poly_space = torch.zeros(1, 1, self.height, self.width, device=transformed_vertices.device)
        
        chunk_size = 512
        
        for i in range(0, self.max_triangles, chunk_size):
            end_i = i + chunk_size
            
            def get_chunk(t):
                # 常に chunk_size (512) のデータを取得する
                chunk = t[:, :, :, i:end_i]
                # 端数の場合、足りない分をゼロパディングする
                if chunk.shape[3] < chunk_size:
                    pad_size = chunk_size - chunk.shape[3]
                    chunk = F.pad(chunk, (0, pad_size), mode='constant', value=0)
                return chunk

            c_A0, c_B0, c_C0 = get_chunk(A0), get_chunk(B0), get_chunk(C0)
            
            def compute_edges(A, B, C):
                weight = torch.cat([A, B, C], dim=1).permute(3, 1, 0, 2).contiguous()
                return F.conv2d(self.pixel_coords, weight, bias=None)

            edges0 = compute_edges(c_A0, c_B0, c_C0)
            edges1 = compute_edges(get_chunk(A1), get_chunk(B1), get_chunk(C1))
            edges2 = compute_edges(get_chunk(A2), get_chunk(B2), get_chunk(C2))

            valid_mask = torch.clamp(torch.relu((c_A0**2 + c_B0**2) * 100.0), min=0.0, max=1.0).permute(3, 1, 0, 2)
            
            inside_cw = torch.relu(edges0 * 100.0) * torch.relu(edges1 * 100.0) * torch.relu(edges2 * 100.0)
            inside_ccw = torch.relu(-edges0 * 100.0) * torch.relu(-edges1 * 100.0) * torch.relu(-edges2 * 100.0)
            
            mask = torch.clamp(torch.maximum(inside_cw, inside_ccw) * valid_mask, min=0.0, max=1.0)

            shading = torch.clamp(get_chunk(poly_normal_Z) * 5.0, min=0.3, max=1.0)
            z_w = torch.clamp(1.0 - (get_chunk(avg_z) / 4.0), min=0.0, max=1.0)
            w = (z_w * shading).permute(3, 1, 0, 2)
            
            chunk_weighted = mask * w
            chunk_max, _ = torch.max(chunk_weighted, dim=1, keepdim=True)
            
            poly_space = torch.maximum(poly_space, chunk_max)

        return poly_space