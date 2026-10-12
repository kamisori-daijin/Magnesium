import asyncio
from pathlib import Path
import numpy as np
import os
import time
from PIL import Image

from coreai.authoring import AIModelAsset
from coreai.runtime import InferenceFunction, NDArray

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
        
        input_mat_name = raytracer_function.desc.input_names[0]
        input_sdf_name = raytracer_function.desc.input_names[1]
        output_port_name = raytracer_function.desc.output_names[0]
        
        num_frames = 60
        print(f"Drawing {num_frames} frames via ANE-Native Pipeline...")
        
        frames_for_gif = []
        start_time = time.time()
        
        for frame in range(num_frames):
            frame_start = time.time()
            angle = (frame / num_frames) * 2.0 * np.pi
            
            cam_x = 2.5 * np.sin(angle)
            cam_y = 1.0
            cam_z = 2.5 * np.cos(angle)
            
            # 転置せずにそのまま flatten
            inv_view_16 = create_inverse_view_matrix(
                eye=[cam_x, cam_y, cam_z],
                target=[0.0, 0.0, 0.0],
                up=[0.0, 1.0, 0.0]
            ).flatten()
            
            # 64要素の配列に、4要素ずつ適切にパッキング
            inv_view_64 = np.zeros(64, dtype=np.float16)
            inv_view_64[0:4]   = inv_view_16[0:4]
            inv_view_64[4:8]   = inv_view_16[4:8]
            inv_view_64[8:12]  = inv_view_16[8:12]
            inv_view_64[12:16] = inv_view_16[12:16]
            
            inv_view_4d_np = inv_view_64.reshape(1, 64, 1, 1)
            
            # --- 動的SDFパラメータの構築 (PyTorch版と完全一致) ---
            sdf_params_np = np.zeros((1, 64, 1, 1), dtype=np.float16)
            
            # 図形1: 中央の大きな球体
            sdf_params_np[0, 0] = 0.0; sdf_params_np[0, 1] = 0.0; sdf_params_np[0, 5] = 0.4
            # 図形2: くり抜く用の動く直方体
            sdf_params_np[0, 8] = 1.0; sdf_params_np[0, 9] = 1.0
            sdf_params_np[0, 10] = 0.5 * np.sin(angle); sdf_params_np[0, 13] = 0.25
            # 図形3: 周回するカプセル
            sdf_params_np[0, 16] = 2.0; sdf_params_np[0, 17] = 0.0
            sdf_params_np[0, 18] = 0.7 * np.cos(angle); sdf_params_np[0, 20] = 0.7 * np.sin(angle); sdf_params_np[0, 21] = 0.15
            # 図形4: 上下するトーラス
            sdf_params_np[0, 24] = 3.0; sdf_params_np[0, 25] = 0.0
            sdf_params_np[0, 27] = 0.5 * np.sin(angle * 3); sdf_params_np[0, 29] = 0.2
            # 図形5: 小さな球（合体）
            sdf_params_np[0, 32] = 0.0; sdf_params_np[0, 33] = 0.0
            sdf_params_np[0, 34] = -0.4; sdf_params_np[0, 35] = 0.4 * np.cos(angle); sdf_params_np[0, 37] = 0.15
            # 図形6: 小さな直方体（くり抜き）
            sdf_params_np[0, 40] = 1.0; sdf_params_np[0, 41] = 1.0
            sdf_params_np[0, 42] = 0.4; sdf_params_np[0, 43] = -0.4; sdf_params_np[0, 45] = 0.15
            # 図形7: 斜めに動くカプセル
            sdf_params_np[0, 48] = 2.0; sdf_params_np[0, 49] = 0.0
            sdf_params_np[0, 50] = 0.3 * np.sin(angle); sdf_params_np[0, 51] = 0.3 * np.sin(angle); sdf_params_np[0, 53] = 0.1
            # 図形8: 中心をくり抜く小さなトーラス
            sdf_params_np[0, 56] = 3.0; sdf_params_np[0, 57] = 1.0
            sdf_params_np[0, 61] = 0.3
            # --------------------------------------------------
            
            inputs = {
                input_mat_name: NDArray(inv_view_4d_np),
                input_sdf_name: NDArray(sdf_params_np)
            }
            
            outputs = await raytracer_function(inputs)
            rendered_output_np = outputs[output_port_name].numpy()
            
            # testmodel.py (PyTorch) の [C, H, W] 形式に合わせる
            img_data = rendered_output_np[0]
            final_img_data = (np.clip(img_data, 0.0, 1.0) * 255).astype(np.uint8)
            
            # PILで保存するために [C, H, W] から [H, W, C] に変換
            final_img_data = np.transpose(final_img_data, (1, 2, 0))
            
            output_filename = f"ane_anim_frames/frame_{frame:03d}.png"
            img = Image.fromarray(final_img_data, 'RGB')
            img.save(output_filename)
            frames_for_gif.append(img)
            
            frame_time = time.time() - frame_start
            fps = 1.0 / frame_time if frame_time > 0 else 0.0
            print(f" Frame {frame+1}/{num_frames} | FPS: {fps:.2f}")
            
        total_time = time.time() - start_time
        avg_fps = num_frames / total_time
        print(f"\nAverage FPS: {avg_fps:.2f}")
        
        # GIFの保存
        gif_path = "ane_anim_frames/output.gif"
        frames_for_gif[0].save(
            gif_path,
            save_all=True,
            append_images=frames_for_gif[1:],
            optimize=False,
            duration=50,
            loop=0
        )
        print(f"Success! GIF saved at: {gif_path}")

if __name__ == "__main__":
    asyncio.run(main())