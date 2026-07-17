import torch
import numpy as np
from PIL import Image

# 100%ANE直撃構造にリファクタリングした2つのモデルをインポート
from ShaderModel import ANE3DRenderer
from MVPProcessor import ANEMVPProcessor

WIDTH = 256
HEIGHT = 256
MAX_VERTICES = 65536
MAX_TRIANGLES = 2000

# 1. 2つの独立したプロセッサモデルを初期化
mvp_processor = ANEMVPProcessor(max_vertices=MAX_VERTICES)
rasterizer = ANE3DRenderer(max_triangles=MAX_TRIANGLES, width=WIDTH, height=HEIGHT)

mvp_processor.eval()
rasterizer.eval()

# -------------------------------------------------------------------------
# 2. カメラ行列（4x4ビューマトリクス）の作成 ★LookAt仕様
# -------------------------------------------------------------------------
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
    
    return torch.from_numpy(R @ T)

# カメラの位置を計算（仰角15度、方位角30度、距離2.0）
distance = 2.0
yaw = np.radians(30.0)
pitch = np.radians(15.0)

cam_x = distance * np.cos(pitch) * np.sin(yaw)
cam_y = distance * np.sin(pitch)
cam_z = distance * np.cos(pitch) * np.cos(yaw)

# 原点 (0,0,0) をロックオンする完璧なカメラ行列を生成
camera_mat = create_camera_matrix(
    eye=[cam_x, cam_y, cam_z], 
    target=[0.0, 0.0, 0.0], 
    up=[0.0, 1.0, 0.0]
)

# -------------------------------------------------------------------------
# 3. テスト用の3Dポリゴンデータの生成（アンインデックス化バッファ）
# -------------------------------------------------------------------------
# 原点付近に、3つの頂点（p0, p1, p2）を持つ綺麗で大きめの正三角形を1枚定義
# 座標: [X, Y, Z, W=1]
p0 = [ 0.0,  0.5, 0.0, 1.0]
p1 = [ 0.5, -0.4, 0.0, 1.0]
p2 = [-0.5, -0.4, 0.0, 1.0]

# 三角形1枚（3頂点分）を配列にする
active_vertices = np.array([p0, p1, p2], dtype=np.float32).T # 形状: (4, 3)

# ANEの固定ポート枠（MAX_VERTICES = 65536）に合わせて残りを0でパディング
padding = np.zeros((4, MAX_VERTICES - 3), dtype=np.float32)
combined_vertices = np.hstack([active_vertices, padding])

# 第1段が要求する完璧な4次元画像レイアウト [1, 4, 1, MAX_VERTICES] に拡張
vertex_buffer = torch.from_numpy(combined_vertices).unsqueeze(0).unsqueeze(2)

# -------------------------------------------------------------------------
# 4. 究極の「2段プレス」パイプラインの実行
# -------------------------------------------------------------------------
with torch.no_grad():
    # 【第1段：MVP】カメラ行列を使って、頂点を一撃で2Dスクリーン座標へ変換
    # 吐き出される形状: [1, 3, 1, MAX_VERTICES]
    transformed_vertices = mvp_processor(camera_mat, vertex_buffer)
    
    # 【第2段：ラスタライザ】変換後のバトンをそのまま直撃させ、画面にピクセルとして現像！
    # 吐き出される形状: [1, 3, H, W]
    output = rasterizer(transformed_vertices)

# -------------------------------------------------------------------------
# 5. 【カラー版】PNG画像として出力
# -------------------------------------------------------------------------
# [1, 3, H, W] -> [H, W, 3] に並び替えて画像化
img_data = output.squeeze(0).permute(1, 2, 0).numpy()
img_data = np.clip(img_data * 255.0, 0, 255).astype(np.uint8)

img = Image.fromarray(img_data, mode='RGB')
img.save("ane_rasterizer_pipeline_test.png")

print("🎉 2段式直撃パイプラインのローカルテストPNG出力が完了しました！")
print("`ane_rasterizer_pipeline_test.png` を開いて、パースに乗って傾いた『パキパキに鋭利な三角形』を確認してください！")
