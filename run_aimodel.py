import asyncio
from pathlib import Path
import numpy as np
from PIL import Image

from coreai.authoring import AIModelAsset
from coreai.runtime import InferenceFunction, NDArray

# カメラを回転させるためのヘルパー関数（Unity互換 R @ T）
def create_camera_matrix(yaw_deg, pitch_deg, tx=0.0, ty=0.0, tz=-2.0):
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    
    # Y軸回転
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R_y = np.array([
        [cos_y,  0.0, sin_y, 0.0],
        [0.0,    1.0, 0.0,   0.0],
        [-sin_y, 0.0, cos_y, 0.0],
        [0.0,    0.0, 0.0,   1.0]
    ], dtype=np.float16)
    
    # X軸回転
    cos_x, sin_x = np.cos(pitch), np.sin(pitch)
    R_x = np.array([
        [1.0, 0.0,   0.0,    0.0],
        [0.0, cos_x, -sin_x, 0.0],
        [0.0, sin_x, cos_x,  0.0],
        [0.0, 0.0,   0.0,    1.0]
    ], dtype=np.float16)
    
    # 平行移動（カメラを引く）
    T = np.array([
        [1.0, 0.0, 0.0, -tx],
        [0.0, 1.0, 0.0, -ty],
        [0.0, 0.0, 1.0, -tz], 
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float16)
    
    # ビュー行列
    return R_x @ R_y @ T

async def main():
    # 前の手順で書き出した3Dアセットを指定
    asset_path = Path("./ane_3d_renderer_universal.aimodel")
    
    if not asset_path.exists():
        print(f"Error: {asset_path} not found. Please compile the model first.")
        return

    print("Loading AIModel Asset...")
    asset = AIModelAsset.load(asset_path)
    
    async with asset.executable() as model:
        function: InferenceFunction = model.load_function("main")
        desc = function.desc

        # -----------------------------------------------------------
        # 1. 外部から流し込む「カメラビュー行列」を生成
        # -----------------------------------------------------------
        # 右に30度回り込み、15度見下ろすカメラ
        camera_matrix_np = create_camera_matrix(yaw_deg=30.0, pitch_deg=15.0, tx=0.0, ty=0.0, tz=-2.0)
        
        # モデルのエクスポート時の名前（例: "camera_matrix" や "x"）を動的に検出し、入力をマッピング
        input_key = desc.input_names[0]
        inputs = {
            input_key: NDArray(camera_matrix_np)
        }

        # -----------------------------------------------------------
        # 2. ANE（Apple Neural Engine）で一撃プレスを実行！
        # -----------------------------------------------------------
        print("🚀 Running 3D Raymarching Inference on ANE...")
        outputs = await function(inputs)
        
        output_key = desc.output_names[0]
        result = outputs[output_key].numpy()

    # -----------------------------------------------------------
    # 3. 3Dフルカラー画像（RGB）として保存
    # -----------------------------------------------------------
    print("📸 Inference completed. Saving as RGB image...")
    
    # 形状 [1, 3, 256, 256] から バッチ(1)を絞り出して [3, 256, 256] に
    img_data = np.squeeze(result, axis=0)
    
    # [3, 256, 256] -> PILが要求する [256, 256, 3]（H, W, C）に並び替え
    if img_data.shape[0] == 3:
        img_data = np.transpose(img_data, (1, 2, 0))

    # 0.0〜1.0 の実数値を 0〜255 (uint8) へクランプして変換
    final_img_data = (np.clip(img_data, 0.0, 1.0) * 255).astype(np.uint8)

    # RGBモードの画像として書き出し
    img = Image.fromarray(final_img_data, 'RGB')
    img.save("coreai_3d_ane_result.png")
    
    print("'coreai_3d_ane_result.png' saved！")

if __name__ == "__main__":
    asyncio.run(main())
