import asyncio
from pathlib import Path
import numpy as np
import os
from PIL import Image

from coreai.authoring import AIModelAsset
from coreai.runtime import InferenceFunction, NDArray

def create_cube_multiview_textures():
    """
    Create cube multiview textures.
    """
    tex = np.zeros((1, 3, 256, 256), dtype=np.float16)
    grid = np.linspace(-1.0, 1.0, 256)
    x, y = np.meshgrid(grid, grid)
    cube_mask = ((x >= -0.4) & (x <= 0.4) & (y >= -0.4) & (y <= 0.4)).astype(np.float16)
    tex[0, 0, :, :] = cube_mask  # Front (X, Y)
    tex[0, 1, :, :] = cube_mask  # Top (X, Z)
    tex[0, 2, :, :] = cube_mask  # Side (Y, Z)
    return tex

def create_inverse_view_matrix(eye, target, up):
    """
    Create inverse view matrix.
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
    return inv_view.astype(np.float16)

def create_inverse_model_matrix(angle, scale_y):
    """
    Object model inverse matrix.
    """
    rot_y = np.eye(4, dtype=np.float32)
    rot_y[0, 0] = np.cos(angle)
    rot_y[0, 2] = -np.sin(angle)
    rot_y[2, 0] = np.sin(angle)
    rot_y[2, 2] = np.cos(angle)
    
    scale = np.eye(4, dtype=np.float32)
    scale[1, 1] = scale_y
    
    trans = np.eye(4, dtype=np.float32)
    trans[0, 3] = -0.6  # X軸に少しずらす
    trans[1, 3] = 0.1   # Y軸に少し上げる
    
    model_matrix = trans @ rot_y @ scale
    inv_model = np.linalg.inv(model_matrix)
    return inv_model.astype(np.float16)

async def main():
    raytracer_path = Path("./ane_raytracer.aimodel")
    
    if not raytracer_path.exists():
        print(f"Error: {raytracer_path} not found. Please run convert.py first.")
        return

    print("Loading AIModelAsset entirely onto ANE...")
    raytracer_asset = AIModelAsset.load(raytracer_path)
    
    os.makedirs("ane_anim_frames", exist_ok=True)
    
    async with raytracer_asset.executable() as raytracer_model:
        raytracer_function: InferenceFunction = raytracer_model.load_function("main")
        
        print("Creating Base Mask...")
        multiview_inputs_np = create_cube_multiview_textures()
        
        input_tex_name = raytracer_function.desc.input_names[0]
        input_mat_name = raytracer_function.desc.input_names[1]
        output_port_name = raytracer_function.desc.output_names[0]
        
        num_frames = 30
        print(f"Drawing {num_frames} frames via ANE-Native Pipeline...")
        
        for frame in range(num_frames):
            angle = (frame / num_frames) * 2.0 * np.pi
            
            cam_x = 3.5 * np.sin(angle)
            cam_y = 1.2 * np.cos(angle * 0.5) 
            cam_z = 3.5 * np.cos(angle)
            
            inv_view_16 = create_inverse_view_matrix(
                eye=[cam_x, cam_y, cam_z],
                target=[0.0, 0.0, 0.0],
                up=[0.0, 1.0, 0.0]
            ).flatten()
            
            obj_rot_angle = angle * 1.5
            obj_scale_y = 1.0 + np.sin(angle * 3.0) * 0.3 
            
            inv_model_16 = create_inverse_model_matrix(
                angle=obj_rot_angle,
                scale_y=obj_scale_y
            ).flatten()
            
            inv_view_64 = np.zeros(64, dtype=np.float16)
            
            # [0〜15ch]: 逆カメラ行列 (Inverse Camera Matrix)
            inv_view_64[:16] = inv_view_16
            
            # [16〜31ch]: 物体1の逆モデル行列 (Object 1 Transform Inverse)
            inv_view_64[16:32] = inv_model_16
            
            # [32〜47ch]: 物体2の逆モデル行列 (Object 2 Transform Inverse)
            # ※ゼロ埋めバグを防ぐため、物体1と同じデータをコピーして割り当て
            inv_view_64[32:48] = inv_model_16
            
            # ==========================================================
            # 🎨 🌟 【アーティスト直感パラメーター制御へのデバッグ割当】
            # ==========================================================
            # 🔷 物体1のマテリアル特性 [48〜50ch]：美しい青い光学ガラス
            inv_view_64[48] = 1.0  # 透明度 (1.0 = 完全シースルー)
            inv_view_64[49] = 0.2  # 反射度 (フチは自動的に100%反射になります)
            inv_view_64[50] = 0.3  # 歪み強度 (0.0~0.5で屈折のグニャリ度をアーティスト調整)
            
            # 🪙 物体2のマテリアル特性 [51〜53ch]：完全な不透明鏡面メタル（金属）
            inv_view_64[51] = 0.0  # 透明度 (0.0 = 不透明固体)
            inv_view_64[52] = 1.0  # 反射度 (1.0 = 鏡のように背景を100%映し出す)
            inv_view_64[53] = 0.0  # 歪み強度 (不透明なため、内部で安全に無視されます)
            
            
            # 🎨 物体1の基本色 [54〜56ch]：透明クリスタルガラス
            inv_view_64[54] = 0.95  # R (限界まで明るくして光を通す)
            inv_view_64[55] = 0.98  # G (ほんのわずかに緑を強くしてガラスの高級感を出す)
            inv_view_64[56] = 1.0   # B

            
            # 🎨 物体2の基本色 [57〜59ch]：メタルの反射ベース色（白・シルバー）
            inv_view_64[57] = 1.0  # R
            inv_view_64[58] = 1.0  # G
            inv_view_64[59] = 1.0  # B
            # ==========================================================
            
            inv_view_4d_np = inv_view_64.reshape(1, 64, 1, 1)
            
            inputs = {
                input_tex_name: NDArray(multiview_inputs_np),
                input_mat_name: NDArray(inv_view_4d_np)
            }
            
            outputs = await raytracer_function(inputs)
            
            rendered_output_np = outputs[output_port_name].numpy()
            
            final_frame_rgb = np.transpose(rendered_output_np[0, :, :, :], (1, 2, 0))
            final_img_data = (np.clip(final_frame_rgb, 0.0, 1.0) * 255).astype(np.uint8)
            
            output_filename = f"ane_anim_frames/frame_{frame:03d}.png"
            Image.fromarray(final_img_data, 'RGB').save(output_filename)
            print(f" Frame {frame+1}/{num_frames} -> {output_filename}")
            
        print("\n" + "="*50)
        print(f"Success! Beautifully morphing frames saved in: `ane_anim_frames/`")
        print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
