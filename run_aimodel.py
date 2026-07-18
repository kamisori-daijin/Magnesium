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
        
        p0 = transformed_vertices[0, :2, 0, 0]
        p1 = transformed_vertices[0, :2, 0, 1]
        p2 = transformed_vertices[0, :2, 0, 2]
        print(f"Transformed P0: {p0}")
        print(f"Transformed P1: {p1}")
        print(f"Transformed P2: {p2}")
        
        def get_edge(p_a, p_b):
            A = p_a[1] - p_b[1]
            B = p_b[0] - p_a[0]
            # 修正：[-1.0, 1.0] の座標系に合わせた C の計算
            C = p_a[0] * p_b[1] - p_a[1] * p_b[0]
            
            # float16のアンダーフローを防ぐために正規化
            length = np.sqrt(A*A + B*B)
            if length > 1e-5:
                A, B, C = A / length, B / length, C / length
                
            return A, B, C

        A0, B0, C0 = get_edge(p0, p1)
        A1, B1, C1 = get_edge(p1, p2)
        A2, B2, C2 = get_edge(p2, p0)
        z_depth = transformed_vertices[0, 2, 0, 0]
        inv_z = 1.0 / z_depth if z_depth != 0 else 1.0
 
        input_names = rast_function.desc.input_names
        rast_inputs = {}
         
        def pack(val):
            t = np.full((1, 1, 1, 64), 0.0, dtype=np.float16)
            t[0, 0, 0, 0] = val
            return NDArray(t)
 
        for name in input_names:
            if "A0" in name: rast_inputs[name] = pack(A0)
            elif "B0" in name: rast_inputs[name] = pack(B0)
            elif "C0" in name: rast_inputs[name] = pack(C0)
            elif "A1" in name: rast_inputs[name] = pack(A1)
            elif "B1" in name: rast_inputs[name] = pack(B1)
            elif "C1" in name: rast_inputs[name] = pack(C1)
            elif "A2" in name: rast_inputs[name] = pack(A2)
            elif "B2" in name: rast_inputs[name] = pack(B2)
            elif "C2" in name: rast_inputs[name] = pack(C2)
            # 色情報（白で描画する場合）
            elif "R" in name or "G" in name or "B" in name:
                rast_inputs[name] = pack(1.0)
            # 深度情報
            elif "Z" in name or "depth" in name.lower():
                rast_inputs[name] = pack(inv_z)
            else:
                rast_inputs[name] = pack(0.0)
        rast_outputs = await rast_function(rast_inputs)
        chunk_result = rast_outputs[rast_function.desc.output_names[0]].numpy()
        
        # ★ デバッグログ：ANEの出力を詳細に確認
        print("="*50)
        print(f"Chunk Output Shape: {chunk_result.shape}")
        print(f"Chunk Output Min: {np.min(chunk_result)}")
        print(f"Chunk Output Max: {np.max(chunk_result)}")
        print(f"Chunk Output Mean: {np.mean(chunk_result)}")
        
        # 0以外の値がどれくらいあるか（描画されているピクセル数）
        non_zero = np.count_nonzero(chunk_result)
        print(f"Non-zero pixels: {non_zero} / {chunk_result.size} ({non_zero/chunk_result.size*100:.2f}%)")
        print("="*50)
        
        if chunk_result.ndim == 4:
            chunk_result = np.max(chunk_result[0], axis=0, keepdims=True)
        
        final_image = np.maximum(final_image, chunk_result)

    # 4. 画像保存
    if final_image.ndim == 4:
        img_data = final_image[0, 0] # 1枚目の画像だけを抽出
    else:
        img_data = final_image
    min_val, max_val = np.min(img_data), np.max(img_data)
    if max_val > min_val:
        img_data = (img_data - min_val) / (max_val - min_val)
        
    final_img_data = (np.clip(img_data, 0.0, 1.0) * 255).astype(np.uint8)
    final_img_data = np.repeat(final_img_data[:, :, np.newaxis], 3, axis=2)
    
    Image.fromarray(final_img_data, 'RGB').save("ane_final_output.png")
    print("✨ 'ane_final_output.png' saved successfully!")

if __name__ == "__main__":
    asyncio.run(main())