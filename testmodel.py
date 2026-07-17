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
xs = [0.0, 0.5, -0.5]  # p0_X, p1_X, p2_X
ys = [0.5, -0.4, -0.4] # p0_Y, p1_Y, p2_Y
zs = [-0.5, -0.5, -0.5] # ★Z座標を0.0から「-0.5」へ引き出す！
ws = [1.0, 1.0, 1.0]   # W成分

active_vertices = np.array([xs, ys, zs, ws], dtype=np.float32)

# 残りを0でパディング
padding = np.zeros((4, MAX_VERTICES - 3), dtype=np.float32)
combined_vertices = np.hstack([active_vertices, padding])

# [1, 4, 1, MAX_VERTICES] に拡張
vertex_buffer = torch.zeros(1, 4, 1, MAX_VERTICES, dtype=torch.float32)

# ② 最初の3つの頂点（p0, p1, p2）の [X, Y, Z, W] を、チャンネル次元（dim=1）に直接カチッと仕込む
# p0: (0.0, 0.5, 0.0, 1.0)
vertex_buffer[0, 0, 0, 0] = 0.0  # X
vertex_buffer[0, 1, 0, 0] = 0.5  # Y
vertex_buffer[0, 2, 0, 0] = 0.0  # Z
vertex_buffer[0, 3, 0, 0] = 1.0  # W

# p1: (0.5, -0.4, 0.0, 1.0)
vertex_buffer[0, 0, 0, 1] = 0.5
vertex_buffer[0, 1, 0, 1] = -0.4
vertex_buffer[0, 2, 0, 1] = 0.0
vertex_buffer[0, 3, 0, 1] = 1.0

# p2: (-0.5, -0.4, 0.0, 1.0)
vertex_buffer[0, 0, 0, 2] = -0.5
vertex_buffer[0, 1, 0, 2] = -0.4
vertex_buffer[0, 2, 0, 2] = 0.0
vertex_buffer[0, 3, 0, 2] = 1.0

# メモリの連続性を完全に保証してロック
vertex_buffer = vertex_buffer.contiguous()
with torch.no_grad():
    # 【第1段：MVP】カメラ行列を使って、頂点を一撃で2Dスクリーン座標へ変換
    transformed_vertices = mvp_processor(camera_mat, vertex_buffer)
    
    # ─── ★ここからデバッグプリント追加 ─────────────────────────────────
    print("\n--- 2段パイプライン 結合デバッグログ ---")
    print(f"[第1段 出力シェイプ]: {list(transformed_vertices.shape)}")
    
    # 最初の三角形（p0, p1, p2）のスクリーン変換後の(X, Y, 深度Z)を生データで表示
    v_data = transformed_vertices[0, :, 0, :3].numpy()
    print(f"[最初の三角形の頂点データ (3ch × 3頂点)]:\n{v_data}")
    # ──────────────────────────────────────────────────────────────────
    
    # 【第2段：ラスタライザ】変換後のバトンをそのまま直撃させ、画面にピクセルとして現像！
    output = rasterizer(transformed_vertices)
    
    # ─── ★出力バッファの状態もチェック ────────────────────────────────
    print(f"[第2段 出力シェイプ]: {list(output.shape)}")
    print(f"[最終ピクセル輝度] max: {output.max().item():.4f}, min: {output.min().item():.4f}\n")
    # ──────────────────────────────────────────────────────────────────

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
