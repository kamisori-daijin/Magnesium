import os
import torch
import numpy as np
import torchvision.utils as vutils
from RayTracingCore import ANEVirtualRTCore

def create_inverse_view_matrix(eye, target, up):
    """
    カメラのビュー逆行列（4x4）を生成
    """
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    
    z_axis = (eye - target) / (np.linalg.norm(eye - target) + 1e-5)
    x_axis = np.cross(up, z_axis) / (np.linalg.norm(np.cross(up, z_axis)) + 1e-5)
    y_axis = np.cross(z_axis, x_axis)
    
    R = np.eye(4, dtype=np.float32)
    R[0, :3] = x_axis
    R[1, :3] = y_axis
    R[2, :3] = z_axis
    
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    
    view_matrix = R @ T
    inv_view = np.linalg.inv(view_matrix)
    return torch.from_numpy(inv_view).float()

def to_ane_12ch(mat4x4, eye_pos):
    """
    【完全座標バグ修正版】
    逆行列によって消失してしまうカメラのリアルな3D位置(eye_pos)を、
    ch 3, 7, 11 に確実かつ明示的にハメ込んでパッキングする。
    """
    ane_packed = torch.zeros(12, dtype=torch.float32)
    
    ane_packed[0] = mat4x4[0, 0]
    ane_packed[1] = mat4x4[0, 1]
    ane_packed[2] = mat4x4[0, 2]
    ane_packed[3] = float(eye_pos[0])
    
    ane_packed[4] = mat4x4[1, 0]
    ane_packed[5] = mat4x4[1, 1]
    ane_packed[6] = mat4x4[1, 2]
    ane_packed[7] = float(eye_pos[1])
    
    ane_packed[8] = mat4x4[2, 0]
    ane_packed[9] = mat4x4[2, 1]
    ane_packed[10] = mat4x4[2, 2]
    ane_packed[11] = float(eye_pos[2])
    
    return ane_packed

def generate_test_scene_geometry():
    """
    【完全並び順修正版】
    モデル側の view(1, 6, B, 1, 1) および view(1, 9, P, 1, 1) の切り出し順に
    完全に一致するように、各チャンネルに要素を明示的にパッキングする。
    """
    max_boxes = 4
    max_polygons = 16

    # --- 1. BVH用 AABB 箱データの作成 ---
    box0_min = [-0.5, -0.5, -0.5]
    box0_max = [ 0.5,  0.5,  0.5]
    
    bvh_boxes_64 = torch.zeros(64, dtype=torch.float32)
    bvh_boxes_64[0 * max_boxes + 0] = box0_min[0]
    bvh_boxes_64[1 * max_boxes + 0] = box0_min[1]
    bvh_boxes_64[2 * max_boxes + 0] = box0_min[2]
    bvh_boxes_64[3 * max_boxes + 0] = box0_max[0]
    bvh_boxes_64[4 * max_boxes + 0] = box0_max[1]
    bvh_boxes_64[5 * max_boxes + 0] = box0_max[2]

    # --- 2. ポリゴン（三角形面）データの作成 ---
    v0 = [-0.4, -0.4,  0.4]
    v1 = [ 0.4, -0.4,  0.4]
    v2 = [-0.4,  0.4,  0.4]
    
    v3 = [ 0.4, -0.4,  0.4]
    v4 = [ 0.4,  0.4,  0.4]
    v5 = [-0.4,  0.4,  0.4]

    poly_vertices_256 = torch.zeros(256, dtype=torch.float32)
    
    # --- 三角形0 ---
    poly_vertices_256[0 * max_polygons + 0] = v0[0]
    poly_vertices_256[1 * max_polygons + 0] = v0[1]
    poly_vertices_256[2 * max_polygons + 0] = v0[2]
    poly_vertices_256[3 * max_polygons + 0] = v1[0]
    poly_vertices_256[4 * max_polygons + 0] = v1[1]
    poly_vertices_256[5 * max_polygons + 0] = v1[2] # ここを修正
    poly_vertices_256[6 * max_polygons + 0] = v2[0]
    poly_vertices_256[7 * max_polygons + 0] = v2[1]
    poly_vertices_256[8 * max_polygons + 0] = v2[2]

    # --- 三角形1 ---
    poly_vertices_256[0 * max_polygons + 1] = v3[0]
    poly_vertices_256[1 * max_polygons + 1] = v3[1]
    poly_vertices_256[2 * max_polygons + 1] = v3[2]
    poly_vertices_256[3 * max_polygons + 1] = v4[0]
    poly_vertices_256[4 * max_polygons + 1] = v4[1]
    poly_vertices_256[5 * max_polygons + 1] = v4[2]
    poly_vertices_256[6 * max_polygons + 1] = v5[0]
    poly_vertices_256[7 * max_polygons + 1] = v5[1]
    poly_vertices_256[8 * max_polygons + 1] = v5[2]

    return bvh_boxes_64, poly_vertices_256

def main():
    print("Starting Virtual RT Core testing script...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"-> Using Device: {device}")

    os.makedirs("anim_frames", exist_ok=True)

    max_boxes = 4
    max_polygons = 16
    model = ANEVirtualRTCore(max_boxes=max_boxes, max_polygons=max_polygons).to(device).half()
    model.eval()

    bvh_boxes_64, poly_vertices_256 = generate_test_scene_geometry()
    bvh_boxes_4d = bvh_boxes_64.view(1, 64, 1, 1).to(device).half()
    poly_vertices_4d = poly_vertices_256.view(1, 256, 1, 1).to(device).half()

    num_frames = 30
    print(f"Rendering {num_frames} frames: Executing Hardware-Level Circuit Simulation...")

    with torch.no_grad():
        for frame in range(num_frames):
            angle = (frame / num_frames) * 2.0 * np.pi

            cam_x = float(2.5 * np.sin(angle))
            cam_y = float(0.8 * np.sin(angle * 0.5))
            cam_z = float(2.5 * np.cos(angle))
            
            inv_view_12 = to_ane_12ch(create_inverse_view_matrix(
                eye=[cam_x, cam_y, cam_z], 
                target=[0.0, 0.0, 0.0], 
                up=[0.0, 1.0, 0.0]
            ), eye_pos=[cam_x, cam_y, cam_z])

            inv_view_64 = torch.zeros(64, dtype=torch.float32)
            inv_view_64[0:12] = inv_view_12
            
            inv_view_4d = inv_view_64.view(1, 64, 1, 1).to(device).half()

            output_buffer = model(inv_view_4d, bvh_boxes_4d, poly_vertices_4d)

            render_mask_2d = output_buffer[:, 1:2, :, :].float().cpu()

            output_image = torch.cat([render_mask_2d, render_mask_2d, render_mask_2d], dim=1)
            output_filename = f"anim_frames/frame_{frame:03d}.png"
            
            vutils.save_image(output_image, output_filename, normalize=False)
            print(f" Frame {frame+1}/{num_frames} Circuit Streamed -> {output_filename}")

        print("\n==================================================")
        print("Success! All hardware-emulated frames written to 'anim_frames/'")
        print("==================================================")
    
if __name__ == "__main__":
    main()