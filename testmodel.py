import torch
import numpy as np
from PIL import Image
from ShaderModel import ANE3DRenderer

STEPS = 128
WIDTH = 256
HEIGHT = 256

# 1. 拡張版1x1 Conv単一パイプラインモデルを初期化
model = ANE3DRenderer(steps=STEPS, width=WIDTH, height=HEIGHT)
model.eval()

# -------------------------------------------------------------------------
# 2. カメラ行列（4x4ビューマトリクス）の作成 ★格闘の末に勝ち取ったLookAt仕様
# -------------------------------------------------------------------------
def create_camera_matrix(eye, target, up):
    """
    eye: カメラの位置 (X, Y, Z)
    target: 注視点 (今回は原点)
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
    
    # ビューの回転成分
    R = np.eye(4, dtype=np.float32)
    R[0, :3] = x_axis
    R[1, :3] = y_axis
    R[2, :3] = z_axis
    
    # ビューの平行移動成分
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    
    # ビュー行列を合成してTorchテンソルで返却
    matrix = torch.from_numpy(R @ T)
    return matrix

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
# 3. 【17枚拡張対応】モデルの features の並びに1対1で対応するウェイト
# -------------------------------------------------------------------------
shape_weights = torch.tensor([
    # 基本の1次・2次項 (計6枚)
    0.0,   # 1. X_prime
    0.2,   # 2. Y_prime (中心を少し上に)
    4.0,   # 3. Z_prime (中心を奥に2.0ずらすための 2 * Z * 2.0 の項)
   -1.0,   # 4. X_prime^2 (マイナスで球体を形作る)
   -1.0,   # 5. Y_prime^2
   -1.0,   # 6. Z_prime^2
   
    # 濃密なsin・cos波成分 (計9枚) ─── まずは0.0で大人しくさせます
    0.0,   # 7. sin(X*2)
    0.0,   # 8. cos(X*2)
    0.0,   # 9. sin(Y*2)
    0.0,   # 10. cos(Y*2)
    0.0,   # 11. sin(Z*2)
    0.0,   # 12. cos(Z*2)
    0.0,   # 13. sin(X*8)
    0.0,   # 14. sin(Y*8)
    0.0,   # 15. sin(Z*8)
    
    # 絶対値のフラクタルノイズ成分 (計2枚)
    0.0,   # 16. abs(sin(X*5))
    0.0    # 17. abs(sin(Y*5))
], dtype=torch.float32)

# レンダリング実行
with torch.no_grad():
    output = model(camera_mat, shape_weights)

# -------------------------------------------------------------------------
# 4. 【カラー版】PNG画像として出力
# -------------------------------------------------------------------------
img_data = output.squeeze(0).permute(1, 2, 0).numpy()
img_data = np.clip(img_data * 255.0, 0, 255).astype(np.uint8)

img = Image.fromarray(img_data, mode='RGB')
img.save("ane_3d_color_test.png")

print("LookAtカメラを組み込んだ最終テスト版のPNG出力が完了しました！")
