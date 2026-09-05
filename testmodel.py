import os
import torch
import numpy as np
import torchvision.utils as vutils
# 🌟 修正：カメラ行列対応版のクラスからインポート
from RayTracingCore import ANERayTracingCore

def create_inverse_view_matrix(eye, target, up):
    """
    Python（CPU/MPS）側でカメラのビュー行列の【逆行列】を正確に計算する関数。
    ANEに余計な inverse 演算をさせないための必須ハックです。
    """
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    
    z_axis = (eye - target) / (np.linalg.norm(eye - target) + 1e-5)
    x_axis = np.cross(up, z_axis) / (np.linalg.norm(np.cross(up, z_axis)) + 1e-5)
    y_axis = np.cross(z_axis, x_axis)
    
    # 通常のビュー行列 R
    R = np.eye(4, dtype=np.float32)
    R[0, :3] = x_axis
    R[1, :3] = y_axis
    R[2, :3] = z_axis
    
    # 通常の並進行列 T
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    
    view_matrix = R @ T
    
    # レイトレーシングに必要なのは「逆行列」
    inv_view = np.linalg.inv(view_matrix)
    return torch.from_numpy(inv_view).float()

def main():
    print("🎬 ANE特化型（数値微分＆4乗ハック）：カメラ旋回レイトレーシングを開始します...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"-> 使用デバイス: {device}")

    # アニメーション保存用のディレクトリ作成
    os.makedirs("anim_frames", exist_ok=True)

    # 1. カメラ対応モデルの初期化
    max_steps = 64
    shadow_steps = 16
    model = ANERayTracingCore(max_steps=max_steps, shadow_steps=shadow_steps).to(device).half()
    model.eval()

    # 2. ベース形状（円柱）の3面図テクスチャを作成 [1, 3, 256, 256]
    dummy_input = torch.zeros(1, 3, 256, 256, dtype=torch.float16, device=device)
    
    y, x = torch.meshgrid(torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256), indexing="ij")
    
    # 3つのチャンネル（正面、真上、真横）すべてに共通の円マスクを描画
    # これにより、3面が交差した空間に完全な球体が彫刻されます
    circle_mask = (x*x + y*y) < 0.35
    circle_mask_half = circle_mask.to(device).half()
    
    dummy_input[0, 0, :, :] = circle_mask_half  # 正面 (X, Y)
    dummy_input[0, 1, :, :] = circle_mask_half  # 真上 (X, Z)
    dummy_input[0, 2, :, :] = circle_mask_half  # 真横 (Y, Z)

    # 3. ぐるぐる回すアニメーションループ（全30フレーム）
    num_frames = 30
    print(f"🚀 カメラを円軌道で回転させながら、{num_frames}フレームを一気に出力します...")
    
    with torch.no_grad():
        for frame in range(num_frames):
            # 時間（フレーム）の経過に合わせて角度を計算（1周 360度）
            angle = (frame / num_frames) * 2.0 * np.pi
            
            # 半径 3.5 の円軌道上をカメラが移動
            cam_x = 3.5 * np.sin(angle)
            cam_y = 1.5 * np.sin(angle * 0.5) # 上下にも少しゆらゆら動かす
            cam_z = 3.5 * np.cos(angle)
            
            inv_view_2d = create_inverse_view_matrix(
                eye=[cam_x, cam_y, cam_z], 
                target=[0.0, 0.0, 0.0], 
                up=[0.0, 1.0, 0.0]
            ).flatten() # 1次元の16要素にする
            
            # 2. ANE専用に64要素のゼロ配列を用意し、先頭にカメラ行列をコピペする
            inv_view_64 = torch.zeros(64, dtype=torch.float32)
            inv_view_64[:16] = inv_view_2d
            
            # 3. 完璧な 4次元・64チャンネル形状にして転送！
            inv_view_4d = inv_view_64.view(1, 64, 1, 1).to(device).half()

            # モデルに流し込む
            output_color = model(dummy_input, inv_view_4d)

            # 画像の保存
            output_image = output_color.float().cpu()
            output_filename = f"anim_frames/frame_{frame:03d}.png"
            vutils.save_image(output_image, output_filename, normalize=False)
            print(f" 🟩 Frame {frame+1}/{num_frames} レンダー完了 -> {output_filename}")

    print(f"\n✨ すべてのフレームが `anim_frames/` フォルダに保存されました！")
    print("4乗ハックによる『バキッとしつつも影が超滑らかな円柱』がカメラワーク付きでヌルヌル動きます。")

if __name__ == "__main__":
    main()
