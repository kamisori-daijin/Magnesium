import torch
import numpy as np
import matplotlib.pyplot as plt

from MVPProcessor import ANEMVPProcessor

# テストする頂点数（1万個の3Dポイントを同時にシミュレーション）
NUM_VERTICES = 10000

# 1. モデルの初期化（最大枠を1万個に設定してインスタンス化）
model = ANEMVPProcessor(max_vertices=NUM_VERTICES)
model.eval()

# -------------------------------------------------------------------------
# 2. カメラ行列（4x4ビューマトリクス）の作成 (Unity互換 R @ T)
# -------------------------------------------------------------------------
def create_camera_matrix(yaw_deg, pitch_deg, tx=0.0, ty=0.0, tz=-2.0):
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    
    # Y軸回転（左右の首振り）
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R_y = np.array([
        [cos_y,  0.0, sin_y, 0.0],
        [0.0,    1.0, 0.0,   0.0],
        [-sin_y, 0.0, cos_y, 0.0],
        [0.0,    0.0, 0.0,   1.0]
    ], dtype=np.float32)
    
    # X軸回転（上下の見下ろし）
    cos_x, sin_x = np.cos(pitch), np.sin(pitch)
    R_x = np.array([
        [1.0, 0.0,   0.0,    0.0],
        [0.0, cos_x, -sin_x, 0.0],
        [0.0, sin_x, cos_x,  0.0],
        [0.0, 0.0,   0.0,    1.0]
    ], dtype=np.float32)
    
    # 平行移動（カメラを配置位置の「逆方向」に移動させることでビュー行列にする）
    T = np.array([
        [1.0, 0.0, 0.0, -tx],
        [0.0, 1.0, 0.0, -ty],
        [0.0, 0.0, 1.0, -tz], 
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    matrix = torch.from_numpy(R_x @ R_y @ T)
    return matrix

# ★【ココを修正】カメラを「手前（tz=-3.5）」に引く。
# これで原点にあるオブジェクトがカメラの正面（プラスZ方向）に綺麗に収まります！
camera_mat = create_camera_matrix(yaw_deg=45.0, pitch_deg=30.0, tx=0.0, ty=0.0, tz=-3.5) 

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
