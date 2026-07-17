import torch
import numpy as np
import matplotlib.pyplot as plt

from MVPProcessor import ANEMVPProcessor

# テストする頂点数（1万個の3Dポイントを同時にシミュレーション）
NUM_VERTICES = 10000

# 1. モデルの初期化（最大枠を1万個に設定してインスタンス化）
model = ANEMVPProcessor(max_vertices=NUM_VERTICES)
model.eval()

def create_camera_matrix(eye, target, up):
    """
    eye: カメラの位置
    target: 注視点（今回は原点）
    up: カメラの上方向
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
    
    # 回転行列
    R = np.eye(4, dtype=np.float32)
    R[0, :3] = x_axis
    R[1, :3] = y_axis
    R[2, :3] = z_axis
    
    # 平行移動行列
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    
    # ビュー行列 = R * T
    matrix = torch.from_numpy(R @ T)
    return matrix

# カメラの位置を計算（仰角30度、方位角45度、距離3.5）
distance = 3.5
yaw = np.radians(45.0)
pitch = np.radians(30.0)

cam_x = distance * np.cos(pitch) * np.sin(yaw)
cam_y = distance * np.sin(pitch)
cam_z = distance * np.cos(pitch) * np.cos(yaw)

# 原点 (0,0,0) を見るようにカメラを設定
camera_mat = create_camera_matrix(
    eye=[cam_x, cam_y, cam_z], 
    target=[0.0, 0.0, 0.0], 
    up=[0.0, 1.0, 0.0]
)

# -------------------------------------------------------------------------
# 3. テスト用の3D頂点データ（生の頂点バッファ）の生成
# -------------------------------------------------------------------------
np.random.seed(42)
raw_xyz = np.random.uniform(-0.8, 0.8, (3, NUM_VERTICES)).astype(np.float32)

raw_w = np.ones((1, NUM_VERTICES), dtype=np.float32)
vertex_buffer_np = np.vstack([raw_xyz, raw_w])[np.newaxis, ...] # 形状: (1, 4, 10000)
vertex_buffer = torch.from_numpy(vertex_buffer_np)

# -------------------------------------------------------------------------
# 4. MVPプロセッサ（einsumハック）を実行！
# -------------------------------------------------------------------------
with torch.no_grad():
    output_buffer = model(camera_mat, vertex_buffer) # 形状: [1, 3, 10000]

# -------------------------------------------------------------------------
# 5. 【超スッキリ】結果をそのまま可視化
# -------------------------------------------------------------------------
transformed_points = output_buffer.squeeze(0).numpy()

screen_x = transformed_points[0] # 0chがすでに画面X
screen_y = transformed_points[1] # 1chがすでに画面Y
depth = transformed_points[2]    # 2chがすでに深度Z

# あとはそのままプロットするだけ！
plt.figure(figsize=(6, 6))
# 綺麗に見えるように、カラーマップの基準を depth に指定
plt.scatter(screen_x, screen_y, s=1, c=depth, cmap='viridis', alpha=0.6)
plt.title(f"ANE MVP Processor Test ({NUM_VERTICES} Vertices)")
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')

output_png = "mvp_processor_test.png"
plt.savefig(output_png, dpi=150)
plt.close()

print(f"内蔵パース割算でのテストが完了しました！画像を確認してください！")
