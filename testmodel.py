import torch
import numpy as np
from PIL import Image

# 💡 ご自身のファイル名（ShaderModel, PreProcessor 等）に合わせてインポートしてください
from ShaderModel import ANE3DRenderer64
from PreProcessor import ANE3DPreProcessor64

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
    
    return torch.from_numpy(R @ T).to(torch.float16)

def create_debug_texture():
    # レンダラーが期待する 64チャンネルのテクスチャを生成
    tex = torch.zeros((1, 64, 256, 256), dtype=torch.float16)
    for y in range(256):
        for x in range(256):
            is_white = ((x // 32) + (y // 32)) % 2 == 0
            color = 1.0 if is_white else 0.0
            tex[0, :, y, x] = color
    return tex

def main():
    print("⏳ Initializing PyTorch Modules...")
    pre_model = ANE3DPreProcessor64().to(dtype=torch.float16).eval()
    rast_model = ANE3DRenderer64(width=256, height=256).to(dtype=torch.float16).eval()

    # -----------------------------------------------------------------
    # 1. データの準備 (最初から 1, 64, ... の新設計形状で Tensor を作成)
    # -----------------------------------------------------------------
    expanded_vertices = torch.zeros((1, 64, 4, 3), dtype=torch.float16)
    mvp_weights = torch.zeros((1, 64, 4, 4), dtype=torch.float16)
    colors_r = torch.zeros((1, 64, 1, 1), dtype=torch.float16)
    colors_g = torch.zeros((1, 64, 1, 1), dtype=torch.float16)
    colors_b = torch.zeros((1, 64, 1, 1), dtype=torch.float16)

    base_mvp = create_camera_matrix([2.0, 2.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    
    pyramid_faces = [
        [[ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, 1.0, 1.0]], # Face0
        [[ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, -1.0, 1.0]], # Face1
        [[ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, -1.0, 1.0]], # Face2
        [[ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]], # Face3
    ]
    pyramid_colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]

    for i in range(4):
        colors_r[0, i, 0, 0] = pyramid_colors[i][0]
        colors_g[0, i, 0, 0] = pyramid_colors[i][1]
        colors_b[0, i, 0, 0] = pyramid_colors[i][2]
        
        mvp_weights[0, i, :, :] = base_mvp
        
        face_tensor = torch.tensor(pyramid_faces[i], dtype=torch.float16) # [3, 4]
        expanded_vertices[0, i, :, :] = face_tensor.t() # [4, 3] に転置して代入

    # 残りの空きスロット(4〜63)に単位行列を詰める
    for i in range(4, 64):
        mvp_weights[0, i, :, :] = torch.eye(4, dtype=torch.float16)

    processed_texture = create_debug_texture()

    # -----------------------------------------------------------------
    # 2. パイプラインのフォワード実行 (純粋なPyTorch演算)
    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    # 2. パイプラインのフォワード実行 (純粋なPyTorch演算)
    # -----------------------------------------------------------------
    print("🚀 Running PreProcessor...")
    with torch.no_grad():
        # プリプロセッサの戻り値（21個のタプル）を受ける
        pre_outputs = pre_model(expanded_vertices, mvp_weights, colors_r, colors_g, colors_b)
        
        # 💡 UV座標のダミー（6個分）をテンソルで用意する
        # 現状はカラーバッファと同じ形状 [1, 64, 1, 1] のゼロテンソルを仮に流用
        dummy_uv = torch.zeros((1, 64, 1, 1), dtype=torch.float16)
        uv_params = (dummy_uv, dummy_uv, dummy_uv, dummy_uv, dummy_uv, dummy_uv)
        
        print("🚀 Running Rasterizer...")
        # 1. プリプロ出力 (21個)
        # 2. 生成したUVダミー (6個)
        # 3. テクスチャ (1個)
        # これらをすべて足して合計28個の引数をレンダラーに綺麗にアンパックして流し込みます
        all_args = pre_outputs + uv_params + (processed_texture,)
        
        R, G, B, mask_w, max_inv_z = rast_model(*all_args)


    # -----------------------------------------------------------------
    # 3. 後処理とNumPy変換、画像保存
    # -----------------------------------------------------------------
    print("📸 Processing outputs and saving image...")
    r_out = R[0, 0, :, :].numpy()
    g_out = G[0, 0, :, :].numpy()
    b_out = B[0, 0, :, :].numpy()
    mask_out = mask_w[0, 0, :, :].numpy()

    print("\n=== 🔍 RASTERIZER DEBUG INFO ===")
    print(f"Mask Max Value (マスクの最大値) : {mask_w.max().item()}") # ❌これが0なら、ポリゴンが1つも描画されていません
    print(f"Edges0 Max/Min (エッジ0の範囲) : {pre_outputs[0].max().item()} / {pre_outputs[0].min().item()}")
    print(f"p0_iz Max Value (深度の逆数)    : {pre_outputs[18].max().item()}") # ❌これが0なら、座標変換自体が失敗しています
    print("================================\n")

    safe_mask = mask_out + 1e-6
    final_r = np.where(mask_out > 0.001, r_out / safe_mask, 0.0)
    final_g = np.where(mask_out > 0.001, g_out / safe_mask, 0.0)
    final_b = np.where(mask_out > 0.001, b_out / safe_mask, 0.0)
    
    final_frame_rgb = np.stack([final_r, final_g, final_b], axis=-1)
    final_img_data = (np.clip(final_frame_rgb, 0.0, 1.0) * 255).astype(np.uint8)
    
    Image.fromarray(final_img_data, 'RGB').save("torch_final_output.png")
    print("✨ 'torch_final_output.png' saved successfully via pure PyTorch!")

if __name__ == "__main__":
    main()
