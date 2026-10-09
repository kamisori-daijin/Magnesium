import os
import torch
import numpy as np
from RayTracingCore import ANEVirtualRTCore

def generate_test_scene_geometry():
    max_boxes = 4
    box0_min, box0_max = [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]
    bvh_boxes_64 = torch.zeros(64, dtype=torch.float32)
    
    # モデル側のスライスに合わせて配置
    # b_min_x: [0:4], b_min_y: [4:8], b_min_z: [8:12]
    # b_max_x: [12:16], b_max_y: [16:20], b_max_z: [20:24]
    bvh_boxes_64[0] = box0_min[0]
    bvh_boxes_64[4] = box0_min[1]
    bvh_boxes_64[8] = box0_min[2]
    bvh_boxes_64[12] = box0_max[0]
    bvh_boxes_64[16] = box0_max[1]
    bvh_boxes_64[20] = box0_max[2]

    v0 = [-0.4, -0.4,  0.0]; v1 = [ 0.4, -0.4,  0.0]; v2 = [-0.4,  0.4,  0.0]
    v3 = [ 0.4, -0.4,  0.0]; v4 = [ 0.4,  0.4,  0.0]; v5 = [-0.4,  0.4,  0.0]

    poly_vertices_256 = torch.zeros(256, dtype=torch.float32)
    P = 16 # max_polygons
    
    # ポリゴン0
    poly_vertices_256[0] = v0[0]; poly_vertices_256[P] = v0[1]; poly_vertices_256[2*P] = v0[2]
    poly_vertices_256[3*P] = v1[0]; poly_vertices_256[4*P] = v1[1]; poly_vertices_256[5*P] = v1[2]
    poly_vertices_256[6*P] = v2[0]; poly_vertices_256[7*P] = v2[1]; poly_vertices_256[8*P] = v2[2]

    # ポリゴン1
    poly_vertices_256[1] = v3[0]; poly_vertices_256[P+1] = v3[1]; poly_vertices_256[2*P+1] = v3[2]
    poly_vertices_256[3*P+1] = v4[0]; poly_vertices_256[4*P+1] = v4[1]; poly_vertices_256[5*P+1] = v4[2]
    poly_vertices_256[6*P+1] = v5[0]; poly_vertices_256[7*P+1] = v5[1]; poly_vertices_256[8*P+1] = v5[2]

    return bvh_boxes_64, poly_vertices_256

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ANEVirtualRTCore(max_boxes=4, max_polygons=16).to(device).half()
    model.eval()

    bvh_boxes_64, poly_vertices_256 = generate_test_scene_geometry()
    bvh_boxes_4d = bvh_boxes_64.view(1, 64, 1, 1).to(device).half()
    poly_vertices_4d = poly_vertices_256.view(1, 256, 1, 1).to(device).half()

    inv_view_64 = torch.zeros(64, dtype=torch.float32)
    inv_view_64[3] = 0.0; inv_view_64[7] = 0.0; inv_view_64[11] = 2.0
    inv_view_64[0] = 1.0; inv_view_64[5] = 1.0; inv_view_64[10] = 1.0
    
    inv_view_4d = inv_view_64.view(1, 64, 1, 1).to(device).half()

    print("\n--- Debug: Ray Info ---")
    print(f"Ray Origin: (0.0, 0.0, 2.0)")
    print(f"Ray Direction: Towards (0.0, 0.0, 0.0)")

    with torch.no_grad():
        output_buffer = model(inv_view_4d, bvh_boxes_4d, poly_vertices_4d)
        
        depth = output_buffer[:, 0:1, :, :]
        render_mask = output_buffer[:, 1:2, :, :]
        
        print("\n--- Debug: Output Analysis ---")
        print(f"Min Depth: {torch.min(depth).item()}")
        print(f"Max Depth: {torch.max(depth).item()}")
        print(f"Hit Pixels: {torch.sum(render_mask > 0.5).item()} / {256*256}")

if __name__ == "__main__":
    main()