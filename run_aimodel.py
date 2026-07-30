import asyncio
from pathlib import Path
import numpy as np
from PIL import Image

from coreai.authoring import AIModelAsset
from coreai.runtime import InferenceFunction, NDArray

def create_camera_matrix(eye, target, up):
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    
    z_axis = (eye - target) / np.linalg.norm(eye - target)
    x_axis = np.cross(up, z_axis) / np.linalg.norm(np.cross(up, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    
    R = np.eye(4, dtype=np.float32)
    R[0, :3] = x_axis; R[1, :3] = y_axis; R[2, :3] = z_axis
    
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    
    return (R @ T).astype(np.float16)

# 🌟 追加：検証用の256x256の市松模様（チェッカーボード）テクスチャ画像を生成
def create_debug_texture():
    tex = np.zeros((1, 3, 256, 256), dtype=np.float16)
    for y in range(256):
        for x in range(256):
            is_white = ((x // 32) + (y // 32)) % 2 == 0
            color = 1.0 if is_white else 0.0
            tex[0, :, y, x] = color # R, G, Bすべて同じ値（白黒）
    return tex

async def main():
    mvp_path = Path("./ane_mvp_processor.aimodel")
    rast_path = Path("./ane_3d_rasterizer_64.aimodel")
    # 🌟 追加：新テクスチャプロセッサモデルのパス
    tex_path = Path("./ane_texture_processor.aimodel")
    
    if not mvp_path.exists() or not rast_path.exists() or not tex_path.exists():
        print("Error: 3 Assets (MVP, Rasterizer, Texture) not found.")
        return

    print("Loading 3 Assets onto ANE...")
    mvp_asset = AIModelAsset.load(mvp_path)
    rast_asset = AIModelAsset.load(rast_path)
    tex_asset = AIModelAsset.load(tex_path) # 🌟
    
    async with mvp_asset.executable() as mvp_model, \
               rast_asset.executable() as rast_model, \
               tex_asset.executable() as tex_model: # 🌟 3つを同時実行化
               
        mvp_function: InferenceFunction = mvp_model.load_function("main")
        rast_function: InferenceFunction = rast_model.load_function("main")
        tex_function: InferenceFunction = tex_model.load_function("main") # 🌟

        # ----------------------------------------------------
        # 🚀 [ステップ 0] テクスチャアライメント化の検証
        # ----------------------------------------------------
        print("🚀 [0/3] Running Texture Processor on ANE...")
        raw_tex_np = create_debug_texture()
        
        # モデル定義の入力名（"raw_image"）で流し込む
        tex_inputs = {"raw_image": NDArray(raw_tex_np)}
        tex_outputs = await tex_function(tex_inputs)
        
        # 1x1 Convで 拡張されたテンソルを取得 [1, 64, 256, 256]
        processed_texture_np = tex_outputs[tex_function.desc.output_names[0]].numpy()
        print(f"   ➔ Texture Aligned Shape: {processed_texture_np.shape}")

        # ----------------------------------------------------
        # 🚀 [ステップ 1] カメラ・幾何計算（MVP）
        # ----------------------------------------------------
        print("🚀 [1/3] Running MVP Transformation on ANE...")
        camera_matrix_np = create_camera_matrix([2.0, 2.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        
        MAX_VERTICES = 65536
        vertex_buffer_np = np.zeros((1, 4, 1, MAX_VERTICES), dtype=np.float16)
        
        vertices_data = [
            [ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, 1.0, 1.0],
            [ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, -1.0, 1.0],
            [ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, -1.0, 1.0],
            [ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0],
        ]
        for i, v in enumerate(vertices_data):
            vertex_buffer_np[0, :, 0, i] = v

        mvp_outputs = await mvp_function({"camera_matrix": NDArray(camera_matrix_np), "vertex_buffer": NDArray(vertex_buffer_np)})
        transformed_vertices = mvp_outputs[mvp_function.desc.output_names[0]].numpy()

        # ----------------------------------------------------
        # 🚀 [ステップ 2] 3Dラスタライズ・テクスチャマッピング
        # ----------------------------------------------------
        print("🚀 [2/3] Running 3D Rasterization with Texture on ANE...")
        
        # 最終出力用バッファ [ポリゴン数(4), チャンネル(4: R,G,B,Mask), H(256), W(256)] の想定
        # もし戻り値の形状が [64, 1, 256, 256] 4組 の場合は後でマージします
        final_frame_rgb = np.zeros((256, 256, 3), dtype=np.float32)
        
        def get_edge(p_a, p_b):
            A = p_a[1] - p_b[1]
            B = p_b[0] - p_a[0]
            C = -(A * p_a[0] + B * p_a[1])
            return A, B, C

        def pack(val):
            t = np.full((1, 1, 1, 64), 0.0, dtype=np.float16)
            t[0, 0, 0, 0] = val
            return NDArray(t)

        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)]
        input_names = rast_function.desc.input_names

        for i in range(4):
            idx = i * 3
            p0 = transformed_vertices[0, :2, 0, idx]
            p1 = transformed_vertices[0, :2, 0, idx+1]
            p2 = transformed_vertices[0, :2, 0, idx+2]
            
            A0, B0, C0 = get_edge(p0, p1)
            A1, B1, C1 = get_edge(p1, p2)
            A2, B2, C2 = get_edge(p2, p0)
            
            z_depth = transformed_vertices[0, 2, 0, idx]
            inv_z = 1.0 / z_depth if z_depth != 0 else 1.0
            
            rast_inputs = {}
            c = colors[i]
            
            # 各引数の自動マッピング処理
            for name in input_names:
                if name == "processed_texture":
                    # 🌟 変換した64chテクスチャテンソルをそのまま丸投げ
                    rast_inputs[name] = NDArray(processed_texture_np)
                elif "_col" in name or name in ["r0", "r1", "r2", "g0", "g1", "g2"]:
                    if "r" in name: rast_inputs[name] = pack(c[0])
                    elif "g" in name: rast_inputs[name] = pack(c[1])
                    elif "b" in name: rast_inputs[name] = pack(c[2])
                    else: rast_inputs[name] = pack(1.0)
                elif name == "a0": rast_inputs[name] = pack(A0)
                elif name == "b0": rast_inputs[name] = pack(B0)
                elif name == "c0": rast_inputs[name] = pack(C0)
                elif name == "a1": rast_inputs[name] = pack(A1)
                elif name == "b1": rast_inputs[name] = pack(B1)
                elif name == "c1": rast_inputs[name] = pack(C1)
                elif name == "a2": rast_inputs[name] = pack(A2)
                elif name == "b2": rast_inputs[name] = pack(B2)
                elif name == "c2": rast_inputs[name] = pack(C2)
                elif "z" in name or "weight" in name: rast_inputs[name] = pack(inv_z)
                else: rast_inputs[name] = pack(0.0)
                    
            # ラスタライザ実行
            rast_outputs = await rast_function(rast_inputs)


            
            # 各チャンネル（R, G, B, Mask）の出力を抽出（0番目の物理チャンネルをデパディング）
            # モデルの実際の出力名（"convolution_3"等）のキーから吸い出す
            out_names = rast_function.desc.output_names
            
            # 形状 から、0chプレーンを抽出
            r_out = rast_outputs[out_names[0]].numpy()[0, 0, :, :] # [256, 256]
            g_out = rast_outputs[out_names[1]].numpy()[0, 0, :, :]
            b_out = rast_outputs[out_names[2]].numpy()[0, 0, :, :]
            mask_out = rast_outputs[out_names[3]].numpy()[0, 0, :, :]


            r_raw_data = rast_outputs[out_names[0]].numpy()
            g_raw_data = rast_outputs[out_names[1]].numpy()
            mask_raw_data = rast_outputs[out_names[3]].numpy()
            
            print(f"--- 📊 ポリゴン [{i}] の ANE 出力生データチェック ---")
            print(f"Rチャンネル   - Max: {np.max(r_raw_data)}, Min: {np.min(r_raw_data)}")
            print(f"Maskチャンネル - Max: {np.max(mask_raw_data)}, Min: {np.min(mask_raw_data)}")
            print(f"エッジ座標サンプル p0: {p0}, inv_z: {inv_z}")
            
            # 🌟 Metalシェーダーと全く同じ「デディバイド処理」を再現して検証
            valid_pixels = mask_out > 0.001
            
            if np.any(valid_pixels):
                # 🌟 修正：分母に極小値（1e-6）を足すことで、ゼロ除算と数値の破綻（NaN）を完全に防ぐ！
                safe_mask = mask_out + 1e-6
                poly_r = np.where(valid_pixels, r_out / safe_mask, 0.0)
                poly_g = np.where(valid_pixels, g_out / safe_mask, 0.0)
                poly_b = np.where(valid_pixels, b_out / safe_mask, 0.0)
                
                # フレームバッファへブレンド（MAX合成）
                poly_rgb = np.stack([poly_r, poly_g, poly_b], axis=-1)
                final_frame_rgb = np.maximum(final_frame_rgb, np.clip(poly_rgb, 0.0, 1.0))

    # ----------------------------------------------------
    # 🚀 [ステップ 3] 画像の保存
    # ----------------------------------------------------
    final_img_data = (final_frame_rgb * 255).astype(np.uint8)
    Image.fromarray(final_img_data, 'RGB').save("ane_final_output.png")
    print("✨ 'ane_final_output.png' saved successfully!")

if __name__ == "__main__":
    asyncio.run(main())
