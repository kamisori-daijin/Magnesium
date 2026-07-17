import asyncio
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from coreai.authoring import AIModelAsset
from coreai.runtime import InferenceFunction, NDArray

# ★【LookAt方式】Siri AIとの格闘から勝ち取った、絶対にズレないカメラ行列生成
def create_camera_matrix(eye, target, up):
    """
    eye: カメラの位置 (X, Y, Z)
    target: 注視点 (今回は原点)
    up: カメラの上方向 [0,1,0]
    """
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    
    # 視線方向（Z軸）
    z_axis = (eye - target)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # 右方向（X軸）
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # 上方向（Y軸）
    y_axis = np.cross(z_axis, x_axis)
    
    # ビューの回転成分
    R = np.eye(4, dtype=np.float32)
    R[0, :3] = x_axis
    R[1, :3] = y_axis
    R[2, :3] = z_axis
    
    # ビューの平行移動成分（カメラ位置の逆ベクトルを仕込む）
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    
    # ANEの大好物である float16 にキャストして行列を合体 (R @ T)
    view_matrix = (R @ T).astype(np.float16)
    return view_matrix

async def main():
    asset_path = Path("./ane_mvp_processor.aimodel")
    
    if not asset_path.exists():
        print(f"Error: {asset_path} not found. Please compile ConvertMVP.py first.")
        return

    print("Loading AIModel Asset onto ANE...")
    asset = AIModelAsset.load(asset_path)
    
    async with asset.executable() as model:
        function: InferenceFunction = model.load_function("main")

        # -----------------------------------------------------------
        # 1. 完璧な LookAt 3Dカメラ位置の計算
        # -----------------------------------------------------------
        distance = 3.5
        yaw = np.radians(45.0)
        pitch = np.radians(30.0)

        cam_x = distance * np.cos(pitch) * np.sin(yaw)
        cam_y = distance * np.sin(pitch)
        cam_z = distance * np.cos(pitch) * np.cos(yaw)

        # 原点 (0,0,0) をロックオンする完璧なビュー行列
        camera_matrix_np = create_camera_matrix(
            eye=[cam_x, cam_y, cam_z], 
            target=[0.0, 0.0, 0.0], 
            up=[0.0, 1.0, 0.0]
        )

        # ② 動的頂点バッファ (最大枠 65536 個の固定ポート)
        MAX_VERTICES = 65536
        NUM_VERTICES = 10000 
        
        np.random.seed(42)
        raw_xyz = np.random.uniform(-0.8, 0.8, (3, NUM_VERTICES)).astype(np.float16)
        raw_w = np.ones((1, NUM_VERTICES), dtype=np.float16)
        
        active_vertices = np.vstack([raw_xyz, raw_w])
        padding = np.zeros((4, MAX_VERTICES - NUM_VERTICES), dtype=np.float16)
        
        vertex_buffer_np = np.hstack([active_vertices, padding])[np.newaxis, ...]

        # ハードコード固定ポート
        inputs = {
            "camera_matrix": NDArray(camera_matrix_np),
            "vertex_buffer": NDArray(vertex_buffer_np)
        }

        # -----------------------------------------------------------
        # 2. ANE（Apple Neural Engine）で一撃プレスを実行！
        # -----------------------------------------------------------
        print(f"🚀 Running LookAt-Vertex Transformation for {NUM_VERTICES} points on ANE...")
        outputs = await function(inputs)
        
        output_key = model.load_function("main").desc.output_names[0]
        result = outputs[output_key].numpy()

    # -----------------------------------------------------------
    # 3. 描画
    # -----------------------------------------------------------
    print("📸 Inference completed. Plotting ANE-rendered 2D points...")
    transformed_points = np.squeeze(result, axis=0)

    screen_x = transformed_points[0, :NUM_VERTICES]
    screen_y = transformed_points[1, :NUM_VERTICES]
    depth = transformed_points[2, :NUM_VERTICES]

    plt.figure(figsize=(6, 6))
    plt.scatter(screen_x, screen_y, s=1, c=depth, cmap='viridis', alpha=0.6)
    plt.title(f"CoreAI ANE LookAt MVP Inference ({NUM_VERTICES} Vertices)")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')

    output_png = "coreai_mvp_ane_result.png"
    plt.savefig(output_png, dpi=150)
    plt.close()
    
    print(f"実機テスト画像 `{output_png}` が保存されました！")

if __name__ == "__main__":
    asyncio.run(main())
