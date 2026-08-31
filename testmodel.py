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
    # 1. データの準備
    # -----------------------------------------------------------------
    expanded_vertices = torch.zeros((1, 64, 4, 4), dtype=torch.float16)
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
        
        # 修正前: expanded_vertices[0, i, :, :] = face_tensor.t()
        # 修正後: 転置せず、最初の3頂点分にそのまま代入
        expanded_vertices[0, i, :3, :] = face_tensor

    for i in range(4, 64):
        mvp_weights[0, i, :, :] = torch.eye(4, dtype=torch.float16)

    processed_texture = create_debug_texture()

    # -----------------------------------------------------------------
    # 2. パイプラインのフォワード実行
    # -----------------------------------------------------------------
    print("🚀 Running PreProcessor...")
    
    # 🔍 追加: プリプロセッサ実行前のデータ確認
    print("\n--- 🔍 [DEBUG] 入力データの確認 ---")
    print("Face 0 の頂点1:", expanded_vertices[0, 0, 1, :])
    print("Face 1 の頂点1:", expanded_vertices[0, 1, 1, :])
    print("-----------------------------------\n")
    
    with torch.no_grad():
        pre_outputs = pre_model(expanded_vertices, mvp_weights, colors_r, colors_g, colors_b)
        
        # 🔍 追加: プリプロセッサ実行後のデータ確認
        print("\n--- 🔍 [DEBUG] 入力データの確認 ---")
        print("Face 0 の頂点1:", expanded_vertices[0, 0, 1, :])
        print("Face 1 の頂点1:", expanded_vertices[0, 1, 1, :])
        print("-----------------------------------\n")
        
        dummy_uv = torch.zeros((1, 64, 1, 1), dtype=torch.float16)
        uv_params = (dummy_uv, dummy_uv, dummy_uv, dummy_uv, dummy_uv, dummy_uv)
        
        print("🚀 Running Rasterizer...")
        all_args = pre_outputs + uv_params + (processed_texture,)
        R, G, B, mask_w, max_inv_z = rast_model(*all_args)

    # -----------------------------------------------------------------
    # 3. デバッグ情報の出力
    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    # 3. デバッグ情報の出力 (犯人を特定するための拡張デバッグ)
    # -----------------------------------------------------------------
    print("\n=== 🔍 DETAILED RASTERIZER DEBUG INFO ===")
    
    for i in range(4):
        iz_max = pre_outputs[18][0, i, 0, 0].item() # p0_iz
        e0_max = pre_outputs[0][0, i, 0, 0].item()  # A0 (Edge0の傾き要素)
        
        # 💡 追加: ラスタライザーに渡っている3つのエッジ関数の基準値をチェック
        # (pre_outputs[0]=A0, pre_outputs[3]=A1, pre_outputs[6]=A2)
        e1_max = pre_outputs[3][0, i, 0, 0].item()  # A1
        e2_max = pre_outputs[6][0, i, 0, 0].item()  # A2
        
        status = "✅ カメラの前にあります" if iz_max > 0 else "❌ 画面外または裏面"
        
        print(f"Face {i} ({['Red', 'Green', 'Blue', 'Yellow'][i]}):")
        print(f"  - Status             : {status}")
        print(f"  - Depth (1/Z)        : {iz_max:.4f}")
        print(f"  - Edge0 (A0) Value   : {e0_max:.4f}")
        print(f"  - Edge1 (A1) Value   : {e1_max:.4f}") # 💡 追加
        print(f"  - Edge2 (A2) Value   : {e2_max:.4f}") # 💡 追加

    print(f"\nTotal Mask Max Value   : {mask_w.max().item()}")
    print("========================================\n")


    # -----------------------------------------------------------------
    # 4. 後処理と画像保存
    # -----------------------------------------------------------------
    print("📸 Processing outputs and saving image...")
    r_out = R[0, 0, :, :].numpy()
    g_out = G[0, 0, :, :].numpy()
    b_out = B[0, 0, :, :].numpy()
    mask_out = mask_w[0, 0, :, :].numpy()

    safe_mask = mask_out + 1e-6
    final_r = np.where(mask_out > 0.001, r_out / safe_mask, 0.0)
    final_g = np.where(mask_out > 0.001, g_out / safe_mask, 0.0)
    final_b = np.where(mask_out > 0.001, b_out / safe_mask, 0.0)
    
    final_frame_rgb = np.stack([final_r, final_g, final_b], axis=-1)
    final_img_data = (np.clip(final_frame_rgb, 0.0, 1.0) * 255).astype(np.uint8)
    
    Image.fromarray(final_img_data, 'RGB').save("torch_final_output.png")
    print("✨ 'torch_final_output.png' saved successfully!")

if __name__ == "__main__":
    main()