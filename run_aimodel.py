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
    # 1. Create Y Axis
    rot_y = np.eye(4, dtype=np.float32)
    rot_y[0, 0] = np.cos(angle)
    rot_y[0, 2] = -np.sin(angle)
    rot_y[2, 0] = np.sin(angle)
    rot_y[2, 2] = np.cos(angle)
    
    # 2. Create Scale Matrix
    scale = np.eye(4, dtype=np.float32)
    scale[1, 1] = scale_y
    
    # 3. Create Translation Matrix
    trans = np.eye(4, dtype=np.float32)
    trans[1, 3] = 0.1
    
    # Compose Model Matrix (Scale -> Rotate -> Translate)
    model_matrix = trans @ rot_y @ scale
    
    # Inverse Model Matrix
    inv_model = np.linalg.inv(model_matrix)
    return inv_model.astype(np.float16)

async def main():
    raytracer_path = Path("./ane_raytracer.aimodel")
    
    if not raytracer_path.exists():
        print(f"Error: {raytracer_path} not found. Please run convert.py first.")
        return

    print("Loading AIModelAsset entirely onto ANE...")
    raytracer_asset = AIModelAsset.load(raytracer_path)
    
    # Save Directory
    os.makedirs("ane_anim_frames", exist_ok=True)
    
    async with raytracer_asset.executable() as raytracer_model:
        raytracer_function: InferenceFunction = raytracer_model.load_function("main")
        
        print("Creating Base Mask...")
        multiview_inputs_np = create_cube_multiview_textures()
        
        # Auto-get Input Port Names
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
            
            # [0〜15ch]: Inverce Camera Matrix
            inv_view_64[:16] = inv_view_16
            
            # [16〜31ch]: Object Transform Inverse Matrix
            inv_view_64[16:32] = inv_model_16
            
            
            inv_view_4d_np = inv_view_64.reshape(1, 64, 1, 1)
            
            
            inputs = {
                input_tex_name: NDArray(multiview_inputs_np),
                input_mat_name: NDArray(inv_view_4d_np)
            }
            
          
            outputs = await raytracer_function(inputs)
            
        
            rendered_output_np = outputs[output_port_name].numpy()
            gray_img_2d = rendered_output_np[0, 0, :, :]
            
            final_frame_rgb = np.stack([gray_img_2d, gray_img_2d, gray_img_2d], axis=-1)
            final_img_data = (np.clip(final_frame_rgb, 0.0, 1.0) * 255).astype(np.uint8)
            
            output_filename = f"ane_anim_frames/frame_{frame:03d}.png"
            Image.fromarray(final_img_data, 'RGB').save(output_filename)
            print(f" Frame {frame+1}/{num_frames} -> {output_filename}")
            
        print("\n" + "="*50)
        print(f"Success! Beautifully morphing frames saved in: `ane_anim_frames/`")
        print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
