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
    tex = np.zeros((1, 64, 256, 256), dtype=np.float16) # 💡新設計: 64チャンネル画像
    for y in range(256):
        for x in range(256):
            is_white = ((x // 32) + (y // 32)) % 2 == 0
            color = 1.0 if is_white else 0.0
            tex[0, :, y, x] = color
    return tex

    # (省略: 前半の関数の定義などはそのまま)

async def main():
    pre_path = Path("./ane_3d_pre_processor_64_optimized.aimodel")
    rast_path = Path("./ane_3d_rasterizer_64_optimized.aimodel")
    tex_path = Path("./ane_texture_processor.aimodel")
    
    if not pre_path.exists() or not rast_path.exists() or not tex_path.exists():
        pre_path = Path("./ane_3d_pre_processor_64.aimodel")
        rast_path = Path("./ane_3d_rasterizer_64.aimodel")

    print("Loading Assets onto ANE sequentially to avoid memory pressure...")
    
    # -----------------------------------------------------------------
    # STAGE 0: Texture Processor (ロード ➔ 実行 ➔ 即解放)
    # -----------------------------------------------------------------
    print("🚀 [0/3] Running Texture Processor on ANE...")
    tex_asset = AIModelAsset.load(tex_path) 
    async with tex_asset.executable() as tex_model:
        tex_function = tex_model.load_function("main")
        raw_tex_np = create_debug_texture()
        tex_inputs = {"raw_image": NDArray(raw_tex_np)}
        tex_outputs = await tex_function(tex_inputs)
        processed_texture_np = tex_outputs[tex_function.desc.output_names[0]].numpy()

    # 💡 完全にインデントの左端を揃えて、次の準備に進めます
    # -----------------------------------------------------------------
    # 入力データの準備
    # -----------------------------------------------------------------
    expanded_vertices_np = np.zeros((1, 64, 4, 3), dtype=np.float16)
    mvp_weights_np = np.zeros((1, 64, 4, 4), dtype=np.float16)
    colors_r_np = np.zeros((1, 64, 1, 1), dtype=np.float16)
    colors_g_np = np.zeros((1, 64, 1, 1), dtype=np.float16)
    colors_b_np = np.zeros((1, 64, 1, 1), dtype=np.float16)

    base_mvp = create_camera_matrix([2.0, 2.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    pyramid_faces = [
        [[ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, 1.0, 1.0]],
        [[ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, -1.0, 1.0]],
        [[ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, -1.0, 1.0]],
        [[ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]],
    ]
    pyramid_colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]

    for i in range(4):
        colors_r_np[0, i, 0, 0] = pyramid_colors[i][0]
        colors_g_np[0, i, 0, 0] = pyramid_colors[i][1]
        colors_b_np[0, i, 0, 0] = pyramid_colors[i][2]
        mvp_weights_np[0, i, :, :] = base_mvp
        face_data = np.array(pyramid_faces[i], dtype=np.float16)
        expanded_vertices_np[0, i, :, :] = face_data.T

    for i in range(4, 64):
        mvp_weights_np[0, i, :, :] = np.eye(4, dtype=np.float16)

    # -----------------------------------------------------------------
    # STAGE 1: 3D PreProcessor (ロード ➔ 実行 ➔ 即解放)
    # -----------------------------------------------------------------
    print("🚀 [1/3] Running 3D PreProcessor on ANE...")
    pre_asset = AIModelAsset.load(pre_path)
    async with pre_asset.executable() as pre_model:
        pre_function = pre_model.load_function("main")
        pre_inputs = {
            "expanded_vertices": NDArray(expanded_vertices_np),
            "mvp_weights": NDArray(mvp_weights_np), 
            "colors_r": NDArray(colors_r_np),
            "colors_g": NDArray(colors_g_np),
            "colors_b": NDArray(colors_b_np)
        }
        pre_outputs_raw = await pre_function(pre_inputs)
        pre_outputs = {key: val.numpy() for key, val in pre_outputs_raw.items()}

    # -----------------------------------------------------------------
    # STAGE 2: 3D Rasterizer (ここで初めてラスタライザをロード)
    # -----------------------------------------------------------------
    print("🚀 [2/3] Running 3D Rasterization with Texture on ANE...")
    rast_asset = AIModelAsset.load(rast_path)
    async with rast_asset.executable() as rast_model:
        rast_function = rast_model.load_function("main")
        
        rast_inputs = {}
        rast_inputs['a0'] = NDArray(pre_outputs['sub_0'])
        rast_inputs['b0'] = NDArray(pre_outputs['sub_1'])
        rast_inputs['c0'] = NDArray(pre_outputs['neg_0'])
        
        rast_inputs['a1'] = NDArray(pre_outputs['sub_2'])
        rast_inputs['b1'] = NDArray(pre_outputs['sub_3'])
        rast_inputs['c1'] = NDArray(pre_outputs['neg_1'])
        
        rast_inputs['a2'] = NDArray(pre_outputs['sub_4'])
        rast_inputs['b2'] = NDArray(pre_outputs['sub_5'])
        rast_inputs['c2'] = NDArray(pre_outputs['neg_2'])
   
        color_r_nd = NDArray(pre_outputs['colors_r'])
        color_g_nd = NDArray(pre_outputs['colors_g'])
        color_b_nd = NDArray(pre_outputs['colors_b'])
        
        rast_inputs['r0'] = color_r_nd; rst_inputs['r1'] = color_r_nd; rst_inputs['r2'] = color_r_nd
        rast_inputs['g0'] = color_g_nd; rst_inputs['g1'] = color_g_nd; rst_inputs['g2'] = color_g_nd
        rast_inputs['b0_col'] = color_b_nd; rst_inputs['b1_col'] = color_b_nd; rst_inputs['b2_col'] = color_b_nd
        
        rast_inputs['p0_iz'] = NDArray(pre_outputs['slice_0'])
        rast_inputs['p1_iz'] = NDArray(pre_outputs['slice_1'])
        rast_inputs['p2_iz'] = NDArray(pre_outputs['slice_2'])
        
        rast_inputs['u0'] = color_r_nd; rst_inputs['v0'] = color_r_nd
        rast_inputs['u1'] = color_r_nd; rst_inputs['v1'] = color_r_nd
        rast_inputs['u2'] = color_r_nd; rst_inputs['v2'] = color_r_nd
        
        rast_inputs["processed_texture"] = NDArray(processed_texture_np)

        rast_outputs = await rast_function(rast_inputs)

        out_names = rast_function.desc.output_names
        r_out = rast_outputs[out_names[0]].numpy()[0, 0, :, :]
        g_out = rast_outputs[out_names[1]].numpy()[0, 0, :, :]
        b_out = rast_outputs[out_names[2]].numpy()[0, 0, :, :]
        mask_out = rast_outputs[out_names[3]].numpy()[0, 0, :, :]

    # -----------------------------------------------------------------
    # [3/3] 後処理と画像保存
    # -----------------------------------------------------------------
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

