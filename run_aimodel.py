import asyncio
from pathlib import Path
import numpy as np
import os
from PIL import Image

from coreai.authoring import AIModelAsset
from coreai.runtime import InferenceFunction, NDArray

def create_cube_multiview_textures():
    """
    キューブのマルチビュー用テクスチャを生成 [1, 3, 256, 256]
    """
    tex = np.zeros((1, 3, 256, 256), dtype=np.float16)
    grid = np.linspace(-1.0, 1.0, 256)
    x, y = np.meshgrid(grid, grid)
    cube_mask = ((x >= -0.4) & (x <= 0.4) & (y >= -0.4) & (y <= 0.4)).astype(np.float16)
    tex[0, 0, :, :] = cube_mask  # Front (X, Y)
    tex[0, 1, :, :] = cube_mask  # Top (X, Z)
    tex[0, 2, :, :] = cube_mask  # Side (Y, Z)
    return tex

def create_inverse_view_matrix(eye, target, up):
    """
    カメラのビュー逆行列を生成
    """
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    
    z_axis = (eye - target) / (np.linalg.norm(eye - target) + 1e-5)
    x_axis = np.cross(up, z_axis) / (np.linalg.norm(np.cross(up, z_axis)) + 1e-5)
    y_axis = np.cross(z_axis, x_axis)
    
    R = np.eye(4, dtype=np.float32)
    R[0, :3] = x_axis
    R[1, :3] = y_axis
    R[2, :3] = z_axis
    
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    
    view_matrix = R @ T
    inv_view = np.linalg.inv(view_matrix)
    return inv_view.astype(np.float32)

def create_inverse_model_matrix(angle, scale_y):
    """
    オブジェクトのモデル逆行列を生成
    """
    # 1. Y軸回転
    rot_y = np.eye(4, dtype=np.float32)
    rot_y[0, 0] = np.cos(angle)
    rot_y[0, 2] = -np.sin(angle)
    rot_y[2, 0] = np.sin(angle)
    rot_y[2, 2] = np.cos(angle)
    
    # 2. スケール行列
    scale = np.eye(4, dtype=np.float32)
    scale[1, 1] = scale_y
    
    # 3. 平行移動行列 (少し上に浮かせる)
    trans = np.eye(4, dtype=np.float32)
    trans[1, 3] = 0.1
    
    # 行列の合成 (Scale -> Rotate -> Translate)
    model_matrix = trans @ rot_y @ scale
    
    # 逆行列の計算
    inv_model = np.linalg.inv(model_matrix)
    return inv_model.astype(np.float32)

def to_ane_12ch(mat4x4):
    """
    ANEの連続スライス(Gather全廃)に最適化するため、
    4x4行列の上3行(3x4 = 12要素)だけを抽出しフラット化するヘルパー
    """
    return mat4x4[:3, :].flatten().astype(np.float16)

async def main():
    raytracer_path = Path("./ane_raytracer.aimodel")
    
    if not raytracer_path.exists():
        print(f"Error: {raytracer_path} not found. Please run convert.py first.")
        return

    print("Loading AIModelAsset entirely onto ANE...")
    raytracer_asset = AIModelAsset.load(raytracer_path)
    
    # 保存用ディレクトリ
    os.makedirs("ane_anim_frames", exist_ok=True)
    
    async with raytracer_asset.executable() as raytracer_model:
        raytracer_function: InferenceFunction = raytracer_model.load_function("main")
        
        print("Creating Base Mask...")
        multiview_inputs_np = create_cube_multiview_textures()
        
        # 入出力ポート名の自動取得
        input_tex_name = raytracer_function.desc.input_names[0]
        input_mat_name = raytracer_function.desc.input_names[1]
        output_port_name = raytracer_function.desc.output_names[0]
        
        num_frames = 30
        print(f"Drawing {num_frames} frames via ANE-Native Pipeline...")
        
        for frame in range(num_frames):
            angle = (frame / num_frames) * 2.0 * np.pi
            
            # 1. カメラ位置の計算（円運動）
            cam_x = 3.5 * np.sin(angle)
            cam_y = 1.2 * np.cos(angle * 0.5) 
            cam_z = 3.5 * np.cos(angle)
            
            # ビュー逆行列を生成し、12要素(3x4)にパック
            inv_view_12 = to_ane_12ch(create_inverse_view_matrix(
                eye=[cam_x, cam_y, cam_z],
                target=[0.0, 0.0, 0.0],
                up=[0.0, 1.0, 0.0]
            ))
            
            # 2. オブジェクトの変形
            obj_rot_angle = angle * 1.5
            obj_scale_y = 1.0 + np.sin(angle * 3.0) * 0.3 
            
            # モデル逆行列を生成し、12要素(3x4)にパック
            inv_model_12 = to_ane_12ch(create_inverse_model_matrix(
                angle=obj_rot_angle,
                scale_y=obj_scale_y
            ))
            
            # 3. 64チャンネルの共通入力バッファ [1, 64, 1, 1] を構築
            inv_view_64 = np.zeros(64, dtype=np.float16)
            
            # 【重要】モデル側のスライス仕様 (0:12, 12:24, 24:36) に完全に一致させて格納
            inv_view_64[0:12] = inv_view_12    # 0〜11ch: カメラ回転・位置 (12ch)
            inv_view_64[12:24] = inv_model_12  # 12〜23ch: オブジェクト1の空間変換 (12ch)
            inv_view_64[24:36] = inv_model_12  # 24〜35ch: オブジェクト2の空間変換 (12ch / 仮で同じものを指定)
            
            inv_view_4d_np = inv_view_64.reshape(1, 64, 1, 1)
            
            # 4. 推論実行
            inputs = {
                input_tex_name: NDArray(multiview_inputs_np),
                input_mat_name: NDArray(inv_view_4d_np)
            }
            
            outputs = await raytracer_function(inputs)
            
            # 5. 【重要】3チャンネルRGBカラー画像の復元処理
            rendered_output_np = outputs[output_port_name].numpy() # 形状: [1, 3, H, W]
            
            # バッチ次元を削除 [3, H, W]
            img_rgb_3d = rendered_output_np[0]
            
            # Pillowが解釈できるように軸を入れ替え [3, H, W] -> [H, W, 3] (RGB)
            final_frame_rgb = np.transpose(img_rgb_3d, (1, 2, 0))
            
            # カラークランプと0-255キャスト
            final_img_data = (np.clip(final_frame_rgb, 0.0, 1.0) * 255).astype(np.uint8)
            
            # 画像保存
            output_filename = f"ane_anim_frames/frame_{frame:03d}.png"
            Image.fromarray(final_img_data, 'RGB').save(output_filename)
            print(f" Frame {frame+1}/{num_frames} -> {output_filename}")
            
        print("\n" + "="*50)
        print(f"Success! Beautifully animated RGB frames saved in: `ane_anim_frames/`")
        print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
