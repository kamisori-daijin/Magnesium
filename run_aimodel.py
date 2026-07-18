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

async def main():
    mvp_path = Path("./ane_mvp_processor.aimodel")
    rast_path = Path("./ane_3d_rasterizer_64.aimodel")
    
    if not mvp_path.exists() or not rast_path.exists():
        print("Error: Assets not found.")
        return

    print("Loading Assets onto ANE...")
    mvp_asset = AIModelAsset.load(mvp_path)
    rast_asset = AIModelAsset.load(rast_path)
    
    async with mvp_asset.executable() as mvp_model, rast_asset.executable() as rast_model:
        mvp_function: InferenceFunction = mvp_model.load_function("main")
        rast_function: InferenceFunction = rast_model.load_function("main")

        # 1. カメラと頂点データの準備
        camera_matrix_np = create_camera_matrix([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        
        MAX_VERTICES = 65536
        vertex_buffer_np = np.zeros((1, 4, 1, MAX_VERTICES), dtype=np.float16)
        
        # テスト用の三角形を1つ設定
        vertex_buffer_np[0, :, 0, 0] = [0.0, 1.0, 0.0, 1.0]
        vertex_buffer_np[0, :, 0, 1] = [-1.0, -1.0, 0.0, 1.0]
        vertex_buffer_np[0, :, 0, 2] = [1.0, -1.0, 0.0, 1.0]

        # 2. 第1段：MVP変換 (ANE)
        print("🚀 [1/2] Running MVP Transformation on ANE...")
        mvp_outputs = await mvp_function({"camera_matrix": NDArray(camera_matrix_np), "vertex_buffer": NDArray(vertex_buffer_np)})
        transformed_vertices = mvp_outputs[mvp_function.desc.output_names[0]].numpy()

        # 3. 第2段：ラスタライズ (ANE)
        print("🚀 [2/2] Running 3D Rasterization on ANE...")
        final_image = np.zeros((1, 1, 256, 256), dtype=np.float16)
        
        ACTIVE_VERTICES = 3
        chunk_triangles = 64
        chunk_vertices = chunk_triangles * 3
        
        input_names = rast_function.desc.input_names
        
        for i in range(0, ACTIVE_VERTICES, chunk_vertices):
            rast_inputs = {}
            for name in input_names:
                # 1.0 ではなく、0.001 のような小さな値を使うことでゼロ除算を防ぐ
                val = 0.001 if "weight" in name else 1.0
                rast_inputs[name] = NDArray(np.full((1, 1, 1, 64), val, dtype=np.float16))
            
            rast_outputs = await rast_function(rast_inputs)
            chunk_result = rast_outputs[rast_function.desc.output_names[0]].numpy()
            
            print(f"Chunk Output Shape: {chunk_result.shape}")
            print(f"Chunk Output Min: {np.min(chunk_result)}, Max: {np.max(chunk_result)}")
            
            if chunk_result.ndim == 4:
                chunk_result = np.max(chunk_result[0], axis=0, keepdims=True)
            
            final_image = np.maximum(final_image, chunk_result)

    # 4. 画像保存
    img_data = final_image[0, 0]
    
    # ★自動正規化（0.0 〜 1.0 の範囲に引き伸ばす）
    min_val = np.min(img_data)
    max_val = np.max(img_data)
    
    if max_val > min_val:
        img_data = (img_data - min_val) / (max_val - min_val)
    else:
        img_data = np.zeros_like(img_data)
        
    print(f"Normalized Image Min: {np.min(img_data)}, Max: {np.max(img_data)}")
    
    final_img_data = (np.clip(img_data, 0.0, 1.0) * 255).astype(np.uint8)
    final_img_data = np.repeat(final_img_data[:, :, np.newaxis], 3, axis=2)
    
    Image.fromarray(final_img_data, 'RGB').save("ane_final_output.png")

if __name__ == "__main__":
    asyncio.run(main())