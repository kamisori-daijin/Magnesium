import asyncio
from pathlib import Path
import numpy as np
import os
from PIL import Image

from coreai.authoring import AIModelAsset
from coreai.runtime import InferenceFunction, NDArray

def create_sdf_params():
    """
    8つの図形パラメータを作成 (64チャンネル)
    """
    sdf_params = np.zeros((1, 64, 1, 1), dtype=np.float16)
    
    # 図形1: 中央の大きな球体
    sdf_params[0, 0, 0, 0] = 0.0  # Type: Sphere
    sdf_params[0, 1, 0, 0] = 0.0  # Op: Union
    sdf_params[0, 5, 0, 0] = 0.4  # Size
    
    # 図形2: くり抜く用の直方体
    sdf_params[0, 8, 0, 0] = 1.0  # Type: Box
    sdf_params[0, 9, 0, 0] = 1.0  # Op: Difference
    sdf_params[0, 10, 0, 0] = 0.2 # X
    sdf_params[0, 13, 0, 0] = 0.25 # Size
    
    # 残りの図形はサイズ0で無効化
    for i in range(2, 8):
        idx = i * 8
        sdf_params[0, idx+1, 0, 0] = 0.0 # Op: Union
        sdf_params[0, idx+5, 0, 0] = 0.0 # Size
        
    return sdf_params

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
    return inv_view.astype(np.float16)

async def main():
    raytracer_path = Path("./ane_raytracer.aimodel")
    
    if not raytracer_path.exists():
        print(f"Error: {raytracer_path} not found.")
        return

    print("Loading AIModelAsset entirely onto ANE...")
    raytracer_asset = AIModelAsset.load(raytracer_path)
    
    os.makedirs("ane_anim_frames", exist_ok=True)
    
    async with raytracer_asset.executable() as raytracer_model:
        raytracer_function: InferenceFunction = raytracer_model.load_function("main")
        
        # 入力名の取得
        input_mat_name = raytracer_function.desc.input_names[0]
        input_sdf_name = raytracer_function.desc.input_names[1]
        output_port_name = raytracer_function.desc.output_names[0]
        
        num_frames = 30
        print(f"Drawing {num_frames} frames via ANE-Native Pipeline...")
        
        for frame in range(num_frames):
            angle = (frame / num_frames) * 2.0 * np.pi
            
            cam_x = 2.5 * np.sin(angle)
            cam_y = 1.0
            cam_z = 2.5 * np.cos(angle)
            
            inv_view_16 = create_inverse_view_matrix(
                eye=[cam_x, cam_y, cam_z],
                target=[0.0, 0.0, 0.0],
                up=[0.0, 1.0, 0.0]
            ).flatten()
            
            inv_view_64 = np.zeros(64, dtype=np.float16)
            inv_view_64[:16] = inv_view_16
            inv_view_4d_np = inv_view_64.reshape(1, 64, 1, 1)
            
            # 動的にパラメータを更新
            sdf_params_np = create_sdf_params()
            sdf_params_np[0, 10, 0, 0] = 0.4 * np.sin(angle) # 直方体を動かす
            
            inputs = {
                input_mat_name: NDArray(inv_view_4d_np),
                input_sdf_name: NDArray(sdf_params_np)
            }
            
            outputs = await raytracer_function(inputs)
            rendered_output_np = outputs[output_port_name].numpy()
            
            img_data = np.transpose(rendered_output_np[0], (1, 2, 0))
            final_img_data = (np.clip(img_data, 0.0, 1.0) * 255).astype(np.uint8)
            
            output_filename = f"ane_anim_frames/frame_{frame:03d}.png"
            Image.fromarray(final_img_data, 'RGB').save(output_filename)
            print(f" Frame {frame+1}/{num_frames} -> {output_filename}")
            
        print("\nSuccess! Frames saved.")

if __name__ == "__main__":
    asyncio.run(main())