import asyncio
from pathlib import Path
import numpy as np
from PIL import Image

from coreai.authoring import AIModelAsset
from coreai.runtime import InferenceFunction, NDArray

def create_camera_matrix(eye, target, up):
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    
    z_axis = (eye - target) / np.linalg.norm(eye - target)
    x_axis = np.cross(up, z_axis) / np.linalg.norm(np.cross(up, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    
    R = np.eye(4, dtype=np.float32)
    R[0, :3] = x_axis; R[1, :3] = y_axis; R[2, :3] = z_axis
    
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    
    return (R @ T).astype(np.float16)

def create_debug_texture():
    tex = np.zeros((1, 3, 256, 256), dtype=np.float16)
    for y in range(256):
        for x in range(256):
            is_white = ((x // 32) + (y // 32)) % 2 == 0
            color = 1.0 if is_white else 0.0
            tex[0, :, y, x] = color
    return tex

async def main():
    # Load the 3 AIModel assets
    pre_path = Path("./ane_3d_pre_processor_64.aimodel")
    rast_path = Path("./ane_3d_rasterizer_64.aimodel")
    tex_path = Path("./ane_texture_processor.aimodel")
    
    if not pre_path.exists() or not rast_path.exists() or not tex_path.exists():
        print("Error: 3 Assets not found.")
        return

    print("Loading 3 Assets onto ANE...")
    pre_asset = AIModelAsset.load(pre_path)
    rast_asset = AIModelAsset.load(rast_path)
    tex_asset = AIModelAsset.load(tex_path) 
    
    async with pre_asset.executable() as pre_model, \
               rast_asset.executable() as rast_model, \
               tex_asset.executable() as tex_model: 
               
        pre_function: InferenceFunction = pre_model.load_function("main")
        rast_function: InferenceFunction = rast_model.load_function("main")
        tex_function: InferenceFunction = tex_model.load_function("main") 

        # -----------------------------------------------------------------
        # [0/3] Texture Processor
        # -----------------------------------------------------------------
        print("🚀 [0/3] Running Texture Processor on ANE...")
        raw_tex_np = create_debug_texture()
        tex_inputs = {"raw_image": NDArray(raw_tex_np)}
        tex_outputs = await tex_function(tex_inputs)
        processed_texture_np = tex_outputs[tex_function.desc.output_names[0]].numpy()

        # 1. Vertex buffer: [1, 4, 3, 64] -> (0,0,0,1) 
        expanded_vertices_np = np.zeros((1, 4, 3, 64), dtype=np.float16)
        expanded_vertices_np[0, 3, :, :] = 1.0  # 全ダミー頂点のWを1.0にする
  
        mvp_weights_np = np.zeros((4, 4, 1, 1), dtype=np.float16)
        
        # 3. Color Buffer: [1, 1, 1, 64] 
        colors_r_np = np.zeros((1, 1, 1, 64), dtype=np.float16)
        colors_g_np = np.zeros((1, 1, 1, 64), dtype=np.float16)
        colors_b_np = np.zeros((1, 1, 1, 64), dtype=np.float16)

    
        base_mvp = create_camera_matrix([2.0, 2.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        
     
        for i in range(4):
            for j in range(4):
                mvp_weights_np[i, j, 0, 0] = base_mvp[i, j]

        pyramid_faces = [
            [[ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, 1.0, 1.0]], # Face0
            [[ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, -1.0, 1.0]], # Face1
            [[ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, -1.0, 1.0]], # Face2
            [[ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]], # Face3
        ]
        pyramid_colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]

        for i in range(4):
          
            colors_r_np[0, 0, 0, i] = pyramid_colors[i][0]
            colors_g_np[0, 0, 0, i] = pyramid_colors[i][1]
            colors_b_np[0, 0, 0, i] = pyramid_colors[i][2]
            
            
            face_data = np.array(pyramid_faces[i], dtype=np.float16).T
            expanded_vertices_np[0, :, :, i] = face_data

    
        print("🚀 [1/3] Running 3D PreProcessor on ANE...")
        pre_inputs = {
            "expanded_vertices": NDArray(expanded_vertices_np),
            "mvp_weights": NDArray(mvp_weights_np), 
            "colors_r": NDArray(colors_r_np),
            "colors_g": NDArray(colors_g_np),
            "colors_b": NDArray(colors_b_np)
        }
        pre_outputs = await pre_function(pre_inputs)

    
        print("🚀 [2/3] Running 3D Rasterization with Texture on ANE...")
        
        rast_inputs = {}
        
        
        rast_inputs['a0'] = pre_outputs['sub']
        rast_inputs['b0'] = pre_outputs['sub_1']
        rast_inputs['c0'] = pre_outputs['neg']
        
        rast_inputs['a1'] = pre_outputs['sub_2']
        rast_inputs['b1'] = pre_outputs['sub_3']
        rast_inputs['c1'] = pre_outputs['neg_1']
        
        rast_inputs['a2'] = pre_outputs['sub_4']
        rast_inputs['b2'] = pre_outputs['sub_5']
        rast_inputs['c2'] = pre_outputs['neg_2']
   
        rast_inputs['r0'] = pre_outputs['colors_r']
        rast_inputs['r1'] = pre_outputs['colors_r']
        rast_inputs['r2'] = pre_outputs['colors_r']
        
        rast_inputs['g0'] = pre_outputs['colors_g']
        rast_inputs['g1'] = pre_outputs['colors_g']
        rast_inputs['g2'] = pre_outputs['colors_g']
        
        rast_inputs['b0_col'] = pre_outputs['colors_b']
        rast_inputs['b1_col'] = pre_outputs['colors_b']
        rast_inputs['b2_col'] = pre_outputs['colors_b']
        
        # Z depth
        rast_inputs['z_weight'] = pre_outputs['slice_10']
        
        # Processed texture
        rast_inputs["processed_texture"] = NDArray(processed_texture_np)

     
        rast_outputs = await rast_function(rast_inputs)

        # -----------------------------------------------------------------
        # save
        # -----------------------------------------------------------------
        out_names = rast_function.desc.output_names
        r_out = rast_outputs[out_names[0]].numpy()[0, 0, :, :]
        g_out = rast_outputs[out_names[1]].numpy()[0, 0, :, :]
        b_out = rast_outputs[out_names[2]].numpy()[0, 0, :, :]
        mask_out = rast_outputs[out_names[3]].numpy()[0, 0, :, :]

        safe_mask = mask_out + 1e-6
        final_r = np.where(mask_out > 0.001, r_out / safe_mask, 0.0)
        final_g = np.where(mask_out > 0.001, g_out / safe_mask, 0.0)
        final_b = np.where(mask_out > 0.001, b_out / safe_mask, 0.0)
        
        final_frame_rgb = np.stack([final_r, final_g, final_b], axis=-1)

        final_img_data = (np.clip(final_frame_rgb, 0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(final_img_data, 'RGB').save("ane_final_output.png")
        print("✨ 'ane_final_output.png' saved successfully!")

if __name__ == "__main__":
    asyncio.run(main())
