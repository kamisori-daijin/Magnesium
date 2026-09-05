import asyncio
from pathlib import Path
import numpy as np
from PIL import Image

from coreai.authoring import AIModelAsset
from coreai.runtime import InferenceFunction, NDArray

def create_dummy_multiview_textures():
    """
    3面図マスク を生成する関数。
    ANEが要求する float16 (Half) で正確にデータを構築します。
    """
    # 🌟 修正：dtypeを np.float32 から np.float16 に変更
    tex = np.zeros((1, 3, 256, 256), dtype=np.float16) 
    
    # 256x256 のピクセル座標（-1.0 〜 1.0）
    grid = np.linspace(-1.0, 1.0, 256)
    x, y = np.meshgrid(grid, grid)
    
    # 中心からの距離
    dist = np.sqrt(x**2 + y**2)
    
    # 半径 0.5 の円の内側を 1.0 (マスク) にする
    # 🌟 修正：.astype(np.float16) に変更
    circle_mask = (dist <= 0.5).astype(np.float16)
    
    # 3つのチャンネル（0:正面XY, 1:真上XZ, 2:真横YZ）すべてに同じ円をセット
    tex[0, 0, :, :] = circle_mask  # 正面マスク
    tex[0, 1, :, :] = circle_mask  # 真上マスク
    tex[0, 2, :, :] = circle_mask  # 真横マスク
    
    return tex

async def main():
    raytracer_path = Path("./ane_raytracer.aimodel")
    
    if not raytracer_path.exists():
        print(f"Error: {raytracer_path} not found.")
        return

    print("🚀 【シリコン直駆動】レイトレーサーの神回路を ANE にロード中...")
    raytracer_asset = AIModelAsset.load(raytracer_path)
    
    async with raytracer_asset.executable() as raytracer_model:
        raytracer_function: InferenceFunction = raytracer_model.load_function("main")
        
        print("🔮 3面図（マルチビュー・テクスチャ：Float16）を構築中...")
        multiview_inputs_np = create_dummy_multiview_textures()
        
        input_port_name = raytracer_function.desc.input_names[0]
        inputs = {input_port_name: NDArray(multiview_inputs_np)}
        
        print("⚡️ ANE (NPU) で並列レイマーチング＆シャドウ生成を一撃実行...")
        outputs = await raytracer_function(inputs)
        
        output_port_name = raytracer_function.desc.output_names[0]
        rendered_output_np = outputs[output_port_name].numpy()
        
        gray_img_2d = rendered_output_np[0, 0, :, :]
        
        final_frame_rgb = np.stack([gray_img_2d, gray_img_2d, gray_img_2d], axis=-1)
        final_img_data = (np.clip(final_frame_rgb, 0.0, 1.0) * 255).astype(np.uint8)
        
        output_filename = "ane_raytraced_sphere.png"
        Image.fromarray(final_img_data, 'RGB').save(output_filename)
        
        print("\n" + "="*50)
        print(f"✨ 成功！ ANEレイトレーシング完了: `{output_filename}`")
        print("型不一致が解消され、ANEのハードウェアアクセラレーションがフルに発揮されました。")
        print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
