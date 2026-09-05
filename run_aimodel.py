import asyncio
from pathlib import Path
import numpy as np
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
    # ANE神回路の簡易法線計算が「球体近似」になっているため、
    # あえてバウンディングボックスに余裕を持たせたサイズにしています。
    cube_mask = ((x >= -0.4) & (x <= 0.4) & (y >= -0.4) & (y <= 0.4)).astype(np.float16)
    
    # 3つのチャンネル（正面、真上、真横）すべてに正方形をセット
    tex[0, 0, :, :] = cube_mask  # 正面 (X, Y)
    tex[0, 1, :, :] = cube_mask  # 真上 (X, Z)
    tex[0, 2, :, :] = cube_mask  # 真横 (Y, Z)
    
    return tex


async def main():
    raytracer_path = Path("./ane_raytracer.aimodel")
    
    if not raytracer_path.exists():
        print(f"Error: {raytracer_path} not found.")
        return

    print("🚀 【形状ハックテスト】円柱モデルを ANE にロード中...")
    raytracer_asset = AIModelAsset.load(raytracer_path)
    
    async with raytracer_asset.executable() as raytracer_model:
        raytracer_function: InferenceFunction = raytracer_model.load_function("main")
        
        print("📐 3面図（円柱ハックマスク）を構築中...")
        multiview_inputs_np = create_cube_multiview_textures()
        
        # 🌟 修正：[0] を追加してリストから最初のポート名を取り出す
        input_port_name = raytracer_function.desc.input_names[0]
        inputs = {input_port_name: NDArray(multiview_inputs_np)}
        
        print("⚡️ ANE で円柱をレイトレーシング中...")
        outputs = await raytracer_function(inputs)
        
        # 🌟 修正：[0] を追加してリストから最初のポート名を取り出す
        output_port_name = raytracer_function.desc.output_names[0]
        rendered_output_np = outputs[output_port_name].numpy()
        
        gray_img_2d = rendered_output_np[0, 0, :, :]
        final_frame_rgb = np.stack([gray_img_2d, gray_img_2d, gray_img_2d], axis=-1)
        final_img_data = (np.clip(final_frame_rgb, 0.0, 1.0) * 255).astype(np.uint8)
        
        output_filename = "ane_raytraced_cylinder.png"
        Image.fromarray(final_img_data, 'RGB').save(output_filename)
        
        print("\n" + "="*50)
        print(f"✨ 成功！ 形状ハック完了: `{output_filename}`")
        print("回路を焼き直すことなく、3面図だけで別の立体が出現しました！")
        print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
