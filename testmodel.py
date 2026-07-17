import torch
import numpy as np
from PIL import Image
from ShaderModel import ANE3DRenderer

STEPS = 128
WIDTH = 256
HEIGHT = 256

# 1. 動的パラメータ対応の3Dカラーレンダラーを初期化
model = ANE3DRenderer(steps=STEPS, width=WIDTH, height=HEIGHT)
model.eval()

def create_camera_matrix(yaw_deg, pitch_deg, tx=0.0, ty=0.0, tz=-2.0):
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R_y = np.array([
        [cos_y,  0.0, sin_y, 0.0],
        [0.0,    1.0, 0.0,   0.0],
        [-sin_y, 0.0, cos_y, 0.0],
        [0.0,    0.0, 0.0,   1.0]
    ], dtype=np.float32)
    
    cos_x, sin_x = np.cos(pitch), np.sin(pitch)
    R_x = np.array([
        [1.0, 0.0,   0.0,    0.0],
        [0.0, cos_x, -sin_x, 0.0],
        [0.0, sin_x, cos_x,  0.0],
        [0.0, 0.0,   0.0,    1.0]
    ], dtype=np.float32)
    
    T = np.array([
        [1.0, 0.0, 0.0, -tx],
        [0.0, 1.0, 0.0, -ty],
        [0.0, 0.0, 1.0, -tz], 
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    matrix = torch.from_numpy(R_x @ R_y @ T)
    return matrix

# 2. カメラ設定（右斜め30度、見下ろし15度、後ろに2.0引く）
camera_mat = create_camera_matrix(yaw_deg=30.0, pitch_deg=15.0, tx=0.0, ty=0.0, tz=-2.0)

# 3. 【追加】配置したい3つの球体のパラメータ (X, Y, Z, 半径)
# 好きな位置や大きさにいつでも自由に変更できます
object_params = torch.tensor([
    [ 0.0,  0.1,  2.0, 0.5], # 1個目: 中央の大きな球体
    [-0.7,  0.2,  2.4, 0.3], # 2個目: 左奥の小さな球体
    [ 0.6, -0.1,  1.7, 0.35],# 3個目: 右手前の球体
], dtype=torch.float32)

# レンダリング実行 (引数としてカメラと物体データを一気に渡す)
with torch.no_grad():
    output = model(camera_mat, object_params) # 形状: [1, 3, H, W]

# -------------------------------------------------------------------------
# 4. 【カラー版】PNG画像として出力
# -------------------------------------------------------------------------
# [1, 3, H, W] -> [H, W, 3] に並び替え
img_data = output.squeeze(0).permute(1, 2, 0).numpy()
img_data = np.clip(img_data * 255.0, 0, 255).astype(np.uint8)

# PILを使ってRGB（カラー）モードで保存
img = Image.fromarray(img_data, mode='RGB')
img.save("ane_3d_color_test.png")

print("おめでとうございます！汎用カラー3D空間のPNG出力が完了しました！")
print("`ane_3d_color_test.png` を開いて、複数の真っ赤な球体が並んでいるか確認してください！")
