import os
import torch
import numpy as np
import torchvision.utils as vutils
# 新しい動的ボクセルコアをインポート
from RayTracingCore import ANEDynamicRayTracingCore

def create_inverse_view_matrix(eye, target, up):
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    
    z_axis = (eye - target) / (np.linalg.norm(eye - target) + 1e-5)
    x_axis = np.cross(up, z_axis) / (np.linalg.norm(np.cross(up, z_axis)) + 1e-5)
    y_axis = np.cross(z_axis, x_axis)
    
    R = np.eye(4, dtype=np.float32)
    R[0, :3] = x_axis; R[1, :3] = y_axis; R[2, :3] = z_axis
    
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    
    view_matrix = R @ T
    inv_view = np.linalg.inv(view_matrix)
    return torch.from_numpy(inv_view).float()

def main():
    print("Starting dynamic voxel ray tracing script...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"-> Using Device: {device}")

    os.makedirs("anim_frames", exist_ok=True)

    grid_size = 8
    # ANEDynamicRayTracingCore を初期化
    model = ANEDynamicRayTracingCore(max_steps=64, grid_size=grid_size).to(device).half()
    model.eval()

    # 【変更点】動的ボクセルデータの作成 (8x8x8 の中心に 4x4x4 の立方体を作る)
    voxel_data = torch.zeros(1, 1, grid_size, grid_size, grid_size, dtype=torch.float16, device=device)
    voxel_data[:, :, 2:6, 2:6, 2:6] = 1.0 # 中心を埋める

    num_frames = 30
    print(f"Rendering {num_frames} frames...")

    with torch.no_grad():
        for frame in range(num_frames):
            angle = (frame / num_frames) * 2.0 * np.pi

            cam_x = 2.5 * np.sin(angle)
            cam_y = 1.0
            cam_z = 2.5 * np.cos(angle)
            
            inv_view_2d = create_inverse_view_matrix(
                eye=[cam_x, cam_y, cam_z], 
                target=[0.0, 0.0, 0.0], 
                up=[0.0, 1.0, 0.0]
            ).flatten() 

            inv_view_64 = torch.zeros(64, dtype=torch.float32)
            inv_view_64[:16] = inv_view_2d
            inv_view_4d = inv_view_64.view(1, 64, 1, 1).to(device).half()

            # 【変更点】voxel_data を渡す
            output_mask = model(voxel_data, inv_view_4d)

            # 白黒画像として保存
            # 【変更】モデルが3チャンネル(RGB)を返すようになったので、そのまま保存します
            output_image = output_mask.float().cpu()
            output_filename = f"anim_frames/frame_{frame:03d}.png"
            vutils.save_image(output_image, output_filename, normalize=False)
            print(f" Frame {frame+1}/{num_frames} Success -> {output_filename}")

        print("All frames rendered successfully!")
    
if __name__ == "__main__":
    main()