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
    
    # 平行移動（カメラを後ろに下げる）
    T = np.array([
        [1.0, 0.0, 0.0, -tx],
        [0.0, 1.0, 0.0, -ty],
        [0.0, 0.0, 1.0, -tz], 
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    matrix = torch.from_numpy(R_x @ R_y @ T)
    return matrix

# 「右斜め45度から回り込み、30度見下ろすカメラ」
camera_mat = create_camera_matrix(yaw_deg=45.0, pitch_deg=30.0, tx=0.0, ty=0.0, tz=-2.5)

# -------------------------------------------------------------------------
# 3. テスト用の3D頂点データ（生の頂点バッファ）の生成
# -------------------------------------------------------------------------
# 空間の [-1.0 〜 1.0] の間にランダムに1万個のXYZ座標をばらまく
np.random.seed(42)
raw_xyz = np.random.uniform(-0.8, 0.8, (3, NUM_VERTICES)).astype(np.float32)

# W成分（1.0）を追加して 4次元ベクトルにする [1, 4, NUM_VERTICES]
raw_w = np.ones((1, NUM_VERTICES), dtype=np.float32)
vertex_buffer_np = np.vstack([raw_xyz, raw_w])[np.newaxis, ...] # 形状: (1, 4, 10000)
vertex_buffer = torch.from_numpy(vertex_buffer_np)

# -------------------------------------------------------------------------
# 4. MVPプロセッサ（einsumハック）を実行！
# -------------------------------------------------------------------------
with torch.no_grad():
    # 1万個の頂点を一撃並列変換
    output_buffer = model(camera_mat, vertex_buffer) # 形状: [1, 4, 10000]

# -------------------------------------------------------------------------
# 5. 結果を可視化（2Dスクリーン座標への投影テスト）
# -------------------------------------------------------------------------
# 変換後のカメラ空間の X, Y, Z を抽出
transformed_points = output_buffer.squeeze(0).numpy()
X_c = transformed_points[0]
Y_c = transformed_points[1]
Z_c = transformed_points[2]

# 【3D透視投影ハック】奥（Z）の座標を使って、XとYに遠近感を適用（パースペクティブ割算）
# Zがマイナスや0に近づいた時のゼロ除算を防ぐ安全対策を挟む
safe_Z = Z_c + 1e-5
screen_x = X_c / safe_Z
screen_y = Y_c / safe_Z

# Matplotlibを使って散布図（ポイントクラウド）として描画
plt.figure(figsize=(6, 6))
plt.scatter(screen_x, screen_y, s=1, c=Z_c, cmap='viridis_r', alpha=0.6)
plt.title(f"ANE MVP Processor Test ({NUM_VERTICES} Vertices)")
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')

# 画像として保存
output_png = "mvp_processor_test.png"
plt.savefig(output_png, dpi=150)
plt.close()

print(f"頂点変換テストが完了しました！")
print(f"1万個の頂点が一瞬で処理され、画像 `{output_png}` にプロットされました。")
print("カメラ視線で立体的に回転したポイントクラウド（手前ほど黄色、奥ほど紫）を確認してください！")
