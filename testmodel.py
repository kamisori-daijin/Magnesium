import torch
import numpy as np
from PIL import Image
from ShaderModel import ANE3DRenderer

STEPS = 128
WIDTH = 256
HEIGHT = 256

# 1. 動的カメラ・128ステップ対応の3Dカラーレンダラーを初期化
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

# 3. レンダリング実行 (引数はカメラ行列のみに変形)
with torch.no_grad():
    output = model(camera_mat) # 形状: [1, 3, H, W]

# -------------------------------------------------------------------------
# 4. 【カラー版】PNG画像として出力
# -------------------------------------------------------------------------
# [1, 3, H, W] -> [H, W, 3] に並び替え
img_data = output.squeeze(0).permute(1, 2, 0).numpy()
img_data = np.clip(img_data * 255.0, 0, 255).astype(np.uint8)

# PILを使ってRGB（カラー）モードで保存
img = Image.fromarray(img_data, mode='RGB')
img.save("ane_3d_color_test.png")

print("おめでとうございます！動的カメラ・128ステップ版のPNG出力が完了しました！")
print("`ane_3d_color_test.png` を開いて、ボケが引き締まった美しい3D空間を確認してください！")
