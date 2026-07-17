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

# 2. カメラ設定
camera_mat = create_camera_matrix(yaw_deg=30.0, pitch_deg=15.0, tx=0.0, ty=0.0, tz=-2.0)

# 3. 【17枚拡張対応】モデルの features の並びに1対1で対応するウェイト
# 次元エラーを完全に黙らせつつ、綺麗な球体を現像する数値の塊です
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

print("17枚の最強基底に適合したPNG出力が完了しました！次元ミスマッチも完全解消です！")
