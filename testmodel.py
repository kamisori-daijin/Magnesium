import os
import torch
import numpy as np
import torchvision.utils as vutils
from RayTracingCore import ANEDynamicSDFCore

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
    print("Starting Dynamic SDF Combination Test...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"-> Using Device: {device}")

    os.makedirs("anim_frames", exist_ok=True)

    # 動的SDFモデルを初期化
    model = ANEDynamicSDFCore(max_steps=16).to(device).half()
    model.eval()

    num_frames = 60
    print(f"Rendering {num_frames} frames with dynamic shapes...")

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

            # --- 動的SDFパラメータの構築 ---
            sdf_params = torch.zeros(1, 64, 1, 1, dtype=torch.float16, device=device)
            
            # 図形1: 中央の大きな球体 (常に表示)
            sdf_params[0, 0] = 0.0  # Type: Sphere
            sdf_params[0, 1] = 0.0  # Op: Union
            sdf_params[0, 5] = 0.5  # Size
            
            # 図形2: 動く直方体 (フレーム30までは合体、以降はくり抜き)
            sdf_params[0, 8] = 1.0  # Type: Box
            sdf_params[0, 9] = 1.0 if frame > 30 else 0.0 # Op: Difference or Union
            sdf_params[0, 10] = 0.4 * np.sin(angle * 2) # X
            sdf_params[0, 11] = 0.4 * np.cos(angle * 2) # Y
            sdf_params[0, 13] = 0.3  # Size
            
            # 図形3: フレーム15から45の間だけ出現する小さな球
            # 図形3: フレーム15から45の間だけ出現する小さな球
            if 15 <= frame <= 45:
                sdf_params[0, 16] = 0.0  # Type: Sphere
                sdf_params[0, 17] = 0.0  # Op: Union
                sdf_params[0, 18] = -0.5 # X
                sdf_params[0, 20] = 0.5  # Z
                sdf_params[0, 21] = 0.2  # Size
            else:
                            # 【修正】サイズを0にして、合体(Union)にすることで無効化
                sdf_params[0, 17] = 0.0  # Op: Union
                sdf_params[0, 21] = 0.0  # Size
                            
                        # 図形4: 未使用
                        # 【修正】こちらもサイズ0、合体(Union)で無効化
            sdf_params[0, 25] = 0.0  # Op: Union
            sdf_params[0, 29] = 0.0  # Size
            # -----------------------------

            output_mask = model(inv_view_4d, sdf_params)

            output_image = output_mask.float().cpu()
            output_filename = f"anim_frames/frame_{frame:03d}.png"
            vutils.save_image(output_image, output_filename, normalize=False)
            print(f" Frame {frame+1}/{num_frames} Success -> {output_filename}")

        print("All dynamic frames rendered successfully!")
    
if __name__ == "__main__":
    main()