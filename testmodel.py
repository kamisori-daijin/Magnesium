import os
import torch
import numpy as np
import torchvision.utils as vutils
from PIL import Image
import resource
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
    print("Starting Ultimate 8-Shape ANE SDF Test...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"-> Using Device: {device}")

    os.makedirs("anim_frames", exist_ok=True)

    model = ANEDynamicSDFCore(max_steps=16).to(device).half()
    model.eval()

    num_frames = 60
    print(f"Rendering {num_frames} frames with 8 dynamic shapes...")

    frames_for_gif = []

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

            # --- 動的SDFパラメータの構築 (8つすべて使用) ---
            sdf_params = torch.zeros(1, 64, 1, 1, dtype=torch.float16, device=device)
            
            # 図形1: 中央の大きな球体 (ベース)
            sdf_params[0, 0] = 0.0; sdf_params[0, 1] = 0.0; sdf_params[0, 5] = 0.4
            
            # 図形2: くり抜く用の動く直方体
            sdf_params[0, 8] = 1.0; sdf_params[0, 9] = 1.0
            sdf_params[0, 10] = 0.5 * np.sin(angle); sdf_params[0, 13] = 0.25
            
            # 図形3: 周回するカプセル
            sdf_params[0, 16] = 2.0; sdf_params[0, 17] = 0.0
            sdf_params[0, 18] = 0.7 * np.cos(angle); sdf_params[0, 20] = 0.7 * np.sin(angle)
            sdf_params[0, 21] = 0.15
            
            # 図形4: 上下するトーラス
            sdf_params[0, 24] = 3.0; sdf_params[0, 25] = 0.0
            sdf_params[0, 27] = 0.5 * np.sin(angle * 3); sdf_params[0, 29] = 0.2
            
            # 図形5: 小さな球（合体）
            sdf_params[0, 32] = 0.0; sdf_params[0, 33] = 0.0
            sdf_params[0, 34] = -0.4; sdf_params[0, 35] = 0.4 * np.cos(angle); sdf_params[0, 37] = 0.15
            
            # 図形6: 小さな直方体（くり抜き）
            sdf_params[0, 40] = 1.0; sdf_params[0, 41] = 1.0
            sdf_params[0, 42] = 0.4; sdf_params[0, 43] = -0.4; sdf_params[0, 45] = 0.15
            
            # 図形7: 斜めに動くカプセル
            sdf_params[0, 48] = 2.0; sdf_params[0, 49] = 0.0
            sdf_params[0, 50] = 0.3 * np.sin(angle); sdf_params[0, 51] = 0.3 * np.sin(angle); sdf_params[0, 53] = 0.1
            
            # 図形8: 中心をくり抜く小さなトーラス
            sdf_params[0, 56] = 3.0; sdf_params[0, 57] = 1.0
            sdf_params[0, 61] = 0.3
            # -----------------------------

            # 推論の実行
            output_mask = model(inv_view_4d, sdf_params)

            if device.type == "mps":
                torch.mps.synchronize()

            # プロセス全体のメモリ使用量(MB)を取得
            usage = resource.getrusage(resource.RUSAGE_SELF)
            mem_used_mb = usage.ru_maxrss / (1024 * 1024)

            output_image = output_mask.float().cpu()
            output_filename = f"anim_frames/frame_{frame:03d}.png"
            vutils.save_image(output_image, output_filename, normalize=False)
            
            frames_for_gif.append(Image.open(output_filename))
            print(f" Frame {frame+1}/{num_frames} Success | Process Memory: {mem_used_mb:.2f} MB")

        gif_path = "anim_frames/output.gif"
        frames_for_gif[0].save(
            gif_path,
            save_all=True,
            append_images=frames_for_gif[1:],
            optimize=False,
            duration=50,
            loop=0
        )
        print(f"\nSuccess! GIF saved at: {gif_path}")
    
if __name__ == "__main__":
    main()