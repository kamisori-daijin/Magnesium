import asyncio
from pathlib import Path
import numpy as np
import os
from PIL import Image

from coreai.authoring import AIModelAsset
from coreai.runtime import InferenceFunction, NDArray

def create_cube_multiview_textures():
    """
    3面図マスクをハックして「立方体（キューブ）」を削り出します。
    """
    tex = np.zeros((1, 3, 256, 256), dtype=np.float16)
    
    # 256x256 のピクセル座標（-1.0 〜 1.0）
    grid = np.linspace(-1.0, 1.0, 256)
    x, y = np.meshgrid(grid, grid)
    
    # 一辺の長さが 0.8（-0.4 〜 0.4）の正方形マスクを生成
    cube_mask = ((x >= -0.4) & (x <= 0.4) & (y >= -0.4) & (y <= 0.4)).astype(np.float16)
    
    # 3つのチャンネル（正面、真上、真横）すべてに正方形をセット
    tex[0, 0, :, :] = cube_mask  # 正面 (X, Y)
    tex[0, 1, :, :] = cube_mask  # 真上 (X, Z)
    tex[0, 2, :, :] = cube_mask  # 真横 (Y, Z)
    
    return tex

def create_inverse_view_matrix(eye, target, up):
    """
    Python（CPU）側でカメラのビュー行列の【逆行列】を正確に計算する関数。
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
    return inv_view.astype(np.float16)

async def main():
    # 🌟 焼き直したカメラ＆自動法線対応のモデルを指定
    raytracer_path = Path("./ane_raytracer.aimodel")
    
    if not raytracer_path.exists():
        print(f"Error: {raytracer_path} not found. convert_cam.py で先にビルドしてください。")
        return

    print("🚀 【シリコン直駆動V2】カメラ行列対応レイトレーサーを ANE にロード中...")
    raytracer_asset = AIModelAsset.load(raytracer_path)
    
    # 連番画像の出力先フォルダ
    os.makedirs("ane_anim_frames", exist_ok=True)
    
    async with raytracer_asset.executable() as raytracer_model:
        raytracer_function: InferenceFunction = raytracer_model.load_function("main")
        
        print("📐 3面図（立方体ハックマスク：Float16）を構築中...")
        multiview_inputs_np = create_cube_multiview_textures()
        
        # モデルのポート名を取得 (1番目がテクスチャ、2番目が行列)
        input_tex_name = raytracer_function.desc.input_names[0]
        input_mat_name = raytracer_function.desc.input_names[1]
        output_port_name = raytracer_function.desc.output_names[0]
        
        num_frames = 30
        print(f"🎬 カメラを旋回させながら {num_frames} フレームのアニメーションを爆速ストリームします...")
        
        for frame in range(num_frames):
            # 円軌道の計算
            angle = (frame / num_frames) * 2.0 * np.pi
            cam_x = 3.5 * np.sin(angle)
            cam_y = 1.2 * np.cos(angle * 0.5)  # 上下にもゆらゆら
            cam_z = 3.5 * np.cos(angle)
            
            # 1. 4x4の逆行列（16個の数値）をフラットに生成
            inv_view_16 = create_inverse_view_matrix(
                eye=[cam_x, cam_y, cam_z],
                target=[0.0, 0.0, 0.0],
                up=[0.0, 1.0, 0.0]
            ).flatten()
            
            # 🌟 2. 【64chアライメントハック】
            # 64要素の空の配列を用意し、先頭16要素にカメラ行列をコピー
            inv_view_64 = np.zeros(64, dtype=np.float16)
            inv_view_64[:16] = inv_view_16
            
            # 🌟 3. ANEが大歓喜する完璧な4次元 [1, 64, 1, 1] に変形！
            inv_view_4d_np = inv_view_64.reshape(1, 64, 1, 1)
            
            # 辞書にパッケージして投入
            inputs = {
                input_tex_name: NDArray(multiview_inputs_np),
                input_mat_name: NDArray(inv_view_4d_np)
            }
            
            # ANE推論（一撃実行）
            outputs = await raytracer_function(inputs)
            
            # レンダリング結果の保存
            rendered_output_np = outputs[output_port_name].numpy()
            gray_img_2d = rendered_output_np[0, 0, :, :]
            
            final_frame_rgb = np.stack([gray_img_2d, gray_img_2d, gray_img_2d], axis=-1)
            final_img_data = (np.clip(final_frame_rgb, 0.0, 1.0) * 255).astype(np.uint8)
            
            output_filename = f"ane_anim_frames/frame_{frame:03d}.png"
            Image.fromarray(final_img_data, 'RGB').save(output_filename)
            print(f" 🟩 Frame {frame+1}/{num_frames} -> {output_filename}")
            
        print("\n" + "="*50)
        print(f"✨ 完璧！！ アニメーション連番の生成が完了しました！: `ane_anim_frames/`")
        print("外見は完璧な4次元(64ch)のフリをしてANEを素通りし、内部で一瞬で定数にバラして使い倒す神システムです。")
        print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
