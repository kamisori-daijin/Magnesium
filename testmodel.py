import os
import torch
import numpy as np
import torchvision.utils as vutils
# クラス名が最終版（ANERayTracingCoreRTFinal）になっていることを想定
from RayTracingCore import ANERayTracingCore

def create_inverse_view_matrix(eye, target, up):
    """
    Inverse view matrix for ray tracing.
    """
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    
    z_axis = (eye - target) / (np.linalg.norm(eye - target) + 1e-5)
    x_axis = np.cross(up, z_axis) / (np.linalg.norm(np.cross(up, z_axis)) + 1e-5)
    y_axis = np.cross(z_axis, x_axis)
    
    # View Row R
    R = np.eye(4, dtype=np.float32)
    R[0, :3] = x_axis
    R[1, :3] = y_axis
    R[2, :3] = z_axis
    
    # Translation matrix T
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    
    view_matrix = R @ T
    inv_view = np.linalg.inv(view_matrix)
    return torch.from_numpy(inv_view).float()

def create_inverse_model_matrix(angle, scale_y):
    """
    🌟 RTコア化：オブジェクトの動きを制御するためのモデル逆行列を計算
    """
    # 1. Y軸回転行列の作成
    rot_y = np.eye(4, dtype=np.float32)
    rot_y[0, 0] = np.cos(angle)
    rot_y[0, 2] = -np.sin(angle)
    rot_y[2, 0] = np.sin(angle)
    rot_y[2, 2] = np.cos(angle)
    
    # 2. スケール行列の作成 (Y軸方向へのビヨーンという伸縮)
    scale = np.eye(4, dtype=np.float32)
    scale[1, 1] = scale_y
    
    # 3. 平行移動行列 (原点から少し上に浮かせる)
    trans = np.eye(4, dtype=np.float32)
    trans[1, 3] = 0.1
    
    # 合成モデル行列 (Scale -> Rotate -> Translate)
    model_matrix = trans @ rot_y @ scale
    
    # レイをオブジェクト空間へねじ曲げるための「逆行列」を計算
    inv_model = np.linalg.inv(model_matrix)
    return torch.from_numpy(inv_model).float()

def main():
    print("Starting ray tracing script...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"-> Using Device: {device}")

    # アニメーションフレーム保存用のディレクトリ作成
    os.makedirs("anim_frames", exist_ok=True)

    # 1. 統合された最新RTコアモデルの初期化
    max_steps = 64
    shadow_steps = 16
    model = ANERayTracingCore(max_steps=max_steps, shadow_steps=shadow_steps).to(device).half()
    model.eval()

    # 2. テクスチャ入力 [1, 3, 256, 256]
    # 🌟 RTコア化により形状はモデル内に焼き込まれたため、入力テクスチャは完全ゼロ（ダミー）でOK！
    dummy_input = torch.zeros(1, 3, 256, 256, dtype=torch.float16, device=device)

    # 3. アニメーションループ (30フレーム)
    num_frames = 30
    print(f"Rendering {num_frames} frames: Camera orbit + Dynamic object transformation...")

    with torch.no_grad():
        for frame in range(num_frames):
            # アニメーション用の媒介変数（角度）
            angle = (frame / num_frames) * 2.0 * np.pi

            # --- 📷 カメラの円軌道位置を計算 ---
            cam_x = 3.5 * np.sin(angle)
            cam_y = 1.5 * np.sin(angle * 0.5) + 0.5
            cam_z = 3.5 * np.cos(angle)
            
            inv_view_2d = create_inverse_view_matrix(
                eye=[cam_x, cam_y, cam_z], 
                target=[0.0, 0.0, 0.0], 
                up=[0.0, 1.0, 0.0]
            ).flatten() # 16要素のフラット配列に展開

            # --- 📦 立方体オブジェクトのアニメーション計算 ---
            # 時間の経過に合わせて、Y軸を中心にグルグル回転しつつ、Y方向にビヨーンと伸縮する
            obj_rot_angle = angle * 1.5
            obj_scale_y = 1.0 + np.sin(angle * 3.0) * 0.3 # 0.7倍 〜 1.3倍にリアルタイム伸縮
            
            inv_model_2d = create_inverse_model_matrix(
                angle=obj_rot_angle,
                scale_y=obj_scale_y
            ).flatten() # 16要素のフラット配列に展開

            # --- 🛠️ 64チャンネルのANE統合入力バッファの構築 ---
            inv_view_64 = torch.zeros(64, dtype=torch.float32)
            
            # [0〜15ch]: カメラの逆行列
            inv_view_64[:16] = inv_view_2d
            
            # [16〜31ch]: 🌟オブジェクト（立方体）のモデル逆行列！
            inv_view_64[16:32] = inv_model_2d
            
            # ANE（CoreML）にそのまま突っ込める 4Dテンソルレイアウトに展開してハーフ精度化
            inv_view_4d = inv_view_64.view(1, 64, 1, 1).to(device).half()

            # --- 🚀 ANEレイトレーシングコア、直撃実行！ ---
            output_color = model(dummy_input, inv_view_4d)

            # 出力画像の保存
            output_image = output_color.float().cpu()
            output_filename = f"anim_frames/frame_{frame:03d}.png"
            vutils.save_image(output_image, output_filename, normalize=False)
            print(f" Frame {frame+1}/{num_frames} Success -> {output_filename}")

        print("🎉 All frames rendered successfully in 'anim_frames/' directory!")
    
if __name__ == "__main__":
    main()
