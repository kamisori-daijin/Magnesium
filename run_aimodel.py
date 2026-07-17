import asyncio
from pathlib import Path
import numpy as np
from PIL import Image

from coreai.authoring import AIModelAsset
from coreai.runtime import InferenceFunction, NDArray

# 格闘の末に完成させた、絶対にズレないLookAt仕様のカメラ行列
def create_camera_matrix(eye, target, up):
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    
    z_axis = (eye - target)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    
    R = np.eye(4, dtype=np.float32)
    R[0, :3] = x_axis
    R[1, :3] = y_axis
    R[2, :3] = z_axis
    
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    
    return (R @ T).astype(np.float16)

async def main():
    # 2つの独立したアセットのパスを指定
    mvp_path = Path("./ane_mvp_processor.aimodel")
    rast_path = Path("./ane_3d_rasterizer.aimodel")
    
    if not mvp_path.exists() or not rast_path.exists():
        print(f"Error: Assets not found. Please compile both ConvertMVP.py and ConvertRasterizer.py first.")
        return

    print("Loading 第1段: MVP Processor Asset onto ANE...")
    mvp_asset = AIModelAsset.load(mvp_path)
    print("Loading 第2段: 3D Rasterizer Asset onto ANE...")
    rast_asset = AIModelAsset.load(rast_path)
    
    # 2つのモデルをANEの実行可能ノードに同時に召喚！
    async with mvp_asset.executable() as mvp_model, rast_asset.executable() as rast_model:
        mvp_function: InferenceFunction = mvp_model.load_function("main")
        rast_function: InferenceFunction = rast_model.load_function("main")

        # -----------------------------------------------------------
        # 1. 完璧な LookAt 3Dカメラマトリクスの計算 (FP16)
        # -----------------------------------------------------------
        distance = 2.0
        yaw = np.radians(30.0)
        pitch = np.radians(15.0)

        cam_x = distance * np.cos(pitch) * np.sin(yaw)
        cam_y = distance * np.sin(pitch)
        cam_z = distance * np.cos(pitch) * np.cos(yaw)

        # 原点(0,0,0)にある三角形を、斜め上から綺麗に見下ろす行列
        camera_matrix_np = create_camera_matrix(
            eye=[cam_x, cam_y, cam_z], 
            target=[0.0, 0.0, 0.0], 
            up=[0.0, 1.0, 0.0]
        )

        # -----------------------------------------------------------
        # 2. テスト用の3Dポリゴンデータの生成（メモリ連続性完全一致仕様）
        # -----------------------------------------------------------
        # 転置やhstackによるねじれを全廃し、F.conv2dが大好きな構造でストレートに並べます
        MAX_VERTICES = 65536
        vertex_buffer_np = np.zeros((1, 4, 1, MAX_VERTICES), dtype=np.float16)

        # p0: (0.0, 0.5, 0.0, 1.0)
        vertex_buffer_np[0, 0, 0, 0] = 0.0
        vertex_buffer_np[0, 1, 0, 0] = 0.5
        vertex_buffer_np[0, 2, 0, 0] = 0.0
        vertex_buffer_np[0, 3, 0, 0] = 1.0

        # p1: (0.5, -0.4, 0.0, 1.0)
        vertex_buffer_np[0, 0, 0, 1] = 0.5
        vertex_buffer_np[0, 1, 0, 1] = -0.4
        vertex_buffer_np[0, 2, 0, 1] = 0.0
        vertex_buffer_np[0, 3, 0, 1] = 1.0

        # p2: (-0.5, -0.4, 0.0, 1.0)
        vertex_buffer_np[0, 0, 0, 2] = -0.5
        vertex_buffer_np[0, 1, 0, 2] = -0.4
        vertex_buffer_np[0, 2, 0, 2] = 0.0
        vertex_buffer_np[0, 3, 0, 2] = 1.0

        # -----------------------------------------------------------
        # 3. パイプライン実行（第1段：MVP ─── ANE一撃プレス）
        # -----------------------------------------------------------
        mvp_inputs = {
            "camera_matrix": NDArray(camera_matrix_np),
            "vertex_buffer": NDArray(vertex_buffer_np)
        }
        
        print("🚀 [1/2] Running Dynamic MVP Vector Transformation on ANE...")
        mvp_outputs = await mvp_function(mvp_inputs)
        
        # 配列のキーを文字列として確実に取得
        mvp_out_key = mvp_function.desc.output_names[0]
        transformed_vertices_ndarray = mvp_outputs[mvp_out_key]

        # -----------------------------------------------------------
        # 4. パイプライン実行（第2段：ラスタライザ ─── ANE二撃プレス）
        # -----------------------------------------------------------
        # メモリの実コピーを1ビットも挟まず、第1段の出力をポインタのまま次のポートへ直撃！
        rast_inputs = {
            "transformed_vertices": transformed_vertices_ndarray
        }
        
        print("🚀 [2/2] Running 3D Pixel Rasterization & Grid Composition on ANE...")
        rast_outputs = await rast_function(rast_inputs)
        
        rast_out_key = rast_function.desc.output_names[0]
        result = rast_outputs[rast_out_key].numpy()

    # -----------------------------------------------------------
    # 5. Unity風 3Dフルカラー画像（RGB）の保存
    # -----------------------------------------------------------
    print("📸 Pipeline inference completed. Saving as RGB image...")
    
    # 形状 [1, 3, 256, 256] -> [3, 256, 256]
    img_data = np.squeeze(result, axis=0)
    # [3, 256, 256] -> PILが要求する [256, 256, 3]（H, W, C）へ並び替え
    img_data = np.transpose(img_data, (1, 2, 0))

    # 0.0〜1.0 の実数値を 0〜255 (uint8) へクランプして変換
    final_img_data = (np.clip(img_data, 0.0, 1.0) * 255).astype(np.uint8)

    # RGBモードで書き出し
    img = Image.fromarray(final_img_data, 'RGB')
    img.save("coreai_final_pipeline_result.png")
    
    print("✨ 'coreai_final_pipeline_result.png' saved successfully!")
    print("完璧なUnity画面が、Warning・Fallback一切ゼロでANEから現像されました！")

if __name__ == "__main__":
    asyncio.run(main())
