import torch
import numpy as np

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

def main():
    print("⏳ Initializing Modules for Analysis...")
    pre_model = ANE3DPreProcessor64().to(dtype=torch.float16).eval()
    rast_model = ANE3DRenderer64(width=256, height=256).to(dtype=torch.float16).eval()

    expanded_vertices = torch.zeros((1, 64, 4, 3), dtype=torch.float16)
    mvp_weights = torch.zeros((1, 64, 4, 4), dtype=torch.float16)
    colors_r = torch.zeros((1, 64, 1, 1), dtype=torch.float16)
    colors_g = torch.zeros((1, 64, 1, 1), dtype=torch.float16)
    colors_b = torch.zeros((1, 64, 1, 1), dtype=torch.float16)

    base_mvp = create_camera_matrix([2.0, 2.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    
    pyramid_faces = [
        [[ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, 1.0, 1.0]],
        [[ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, -1.0, 1.0]],
        [[ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, -1.0, 1.0]],
        [[ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]],
    ]
    pyramid_colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]

    for i in range(4):
        colors_r[0, i, 0, 0] = pyramid_colors[i][0]
        colors_g[0, i, 0, 0] = pyramid_colors[i][1]
        colors_b[0, i, 0, 0] = pyramid_colors[i][2]
        mvp_weights[0, i, :, :] = base_mvp
        face_tensor = torch.tensor(pyramid_faces[i], dtype=torch.float16)
        expanded_vertices[0, i, :, :] = face_tensor.t()

    for i in range(4, 64):
        mvp_weights[0, i, :, :] = torch.eye(4, dtype=torch.float16)

    processed_texture = torch.zeros((1, 64, 256, 256), dtype=torch.float16)

    print("\n==================================================")
    print(" 🚀 DETAILED PIPELINE DEBUGGING (PyTorch)")
    print("==================================================")
    
    with torch.no_grad():
        transformed = torch.matmul(mvp_weights.to(torch.float16), expanded_vertices.to(torch.float16))
        print(f"[A] Transformed Raw Shape: {list(transformed.shape)}")
        print(f"    Transformed Max/Min  : {transformed.max().item()} / {transformed.min().item()}")
        print(f"    Transformed NaN Count: {torch.isnan(transformed).sum().item()}")

        X_c = transformed[:, :, 0:1, :] 
        Y_c = transformed[:, :, 1:2, :] 
        W_c = transformed[:, :, 3:4, :] 
        print(f"[B] W_c Raw Max/Min       : {W_c.max().item()} / {W_c.min().item()}")
        
        safe_W = torch.abs(W_c) + 1e-5
        screen_x = X_c / safe_W  
        screen_y = Y_c / safe_W  
        inv_Z = 1.0 / safe_W     
        print(f"[C] inv_Z (1/W) Max/Min  : {inv_Z.max().item()} / {inv_Z.min().item()}")
        print(f"    inv_Z IsInf Count    : {torch.isinf(inv_Z).sum().item()}")

        print("\n--- Running Full PreProcessor Module ---")
        pre_outputs = pre_model(expanded_vertices, mvp_weights, colors_r, colors_g, colors_b)
        print(f"    PreProcessor Returned Tuple Length: {len(pre_outputs)}")
        
        param_names = [
            "A0", "B0", "C0", "A1", "B1", "C1", "A2", "B2", "C2",
            "R0", "G0", "B0_col", "R1", "G1", "B1_col", "R2", "G2", "B2_col",
            "p0_iz", "p1_iz", "p2_iz"
        ]
        print("\n--- PreProcessor Output Tuple Inspection ---")
        for idx, name in enumerate(param_names):
            if idx < len(pre_outputs):
                t = pre_outputs[idx]
                # 💡 修正: list(t.shape) を文字列に変形して TypeError を回避！
                shape_str = str(list(t.shape))
                print(f"  [{idx:02d}] {name:<7} -> Shape: {shape_str:<15} | Max: {t.max().item():<8} | Min: {t.min().item():<8} | HasInf: {torch.isinf(t).any().item()}")
            else:
                print(f"  [{idx:02d}] {name:<7} -> ❌ MISSING IN TUPLE")

        dummy_uv = torch.zeros((1, 64, 1, 1), dtype=torch.float16)
        uv_params = (dummy_uv, dummy_uv, dummy_uv, dummy_uv, dummy_uv, dummy_uv)
        all_args = pre_outputs + uv_params + (processed_texture,)
        
        print("\n--- Running Full Rasterizer Module ---")
        R, G, B, mask_w, max_inv_z = rast_model(*all_args)

         # debug_pipeline.py の「--- Running Full Rasterizer Module ---」部分を以下に置き換え
        print("\n==================================================")
        print(" 🔍 RASTERIZER INTERNAL TRACE (ログ増量版)")
        print("==================================================")
        
        # 1. 引数の分解（21個のプリプロ出力 + 6個のUVダミー + 1個のテクスチャ）
        # forwardの引数定義順に合わせて綺麗にアンパック
        A0, B0, C0, A1, B1, C1, A2, B2, C2 = all_args[0:9]
        R0, G0, B0_col, R1, G1, B1_col, R2, G2, B2_col = all_args[9:18]
        p0_iz, p1_iz, p2_iz = all_args[18:21]
        U0, V0, U1, V1, U2, V2 = all_args[21:27]
        processed_texture = all_args[27]
 
        # 💡 レンダラーのforward内部の計算を1行ずつ実行し、NaNの発生箇所を暴く
        edges0 = A0 * rast_model.x_coords + B0 * rast_model.y_coords + C0
        edges1 = A1 * rast_model.x_coords + B1 * rast_model.y_coords + C1
        edges2 = A2 * rast_model.x_coords + B2 * rast_model.y_coords + C2
        print(f"  [Step 1] Edges0 HasNaN        : {torch.isnan(edges0).any().item()}")
        print(f"           Edges0 Max/Min       : {edges0.max().item()} / {edges0.min().item()}")
        valid_mask = torch.clamp((A0 * A0 + B0 * B0) * 100.0, min=0.0, max=1.0)
        mask = torch.relu(edges0 * 100.0)
        mask = mask * torch.relu(edges1 * 100.0)
        mask = mask * torch.relu(edges2 * 100.0)
        mask = mask * valid_mask
        mask = torch.clamp(mask, min=0.0, max=1.0)
        print(f"  [Step 2] Raster Mask HasNaN   : {torch.isnan(mask).any().item()}")
        print(f"           Mask Max/Min         : {mask.max().item()} / {mask.min().item()}")
 
        # 重心座標ガード計算
        total_area = torch.abs(edges0 + edges1 + edges2) + 0.02
        inv_area = torch.reciprocal(total_area)
        print(f"  [Step 3] inv_area HasNaN      : {torch.isnan(inv_area).any().item()}")
        print(f"           inv_area Max/Min     : {inv_area.max().item()} / {inv_area.min().item()}")
        print(f"           inv_area HasInf      : {torch.isinf(inv_area).any().item()}")

        w0 = edges1 * inv_area
        w1 = edges2 * inv_area
        w2 = edges0 * inv_area
        print(f"  [Step 4] Weights (w0) HasNaN  : {torch.isnan(w0).any().item()}")
        pixel_inv_z = (p0_iz * w0) + (p1_iz * w1) + (p2_iz * w2)
        pixel_inv_z = pixel_inv_z * mask
        print(f"  [Step 5] Pixel Inv Z HasNaN   : {torch.isnan(pixel_inv_z).any().item()}")

        u_interp = (U0 * w0) + (U1 * w1) + (U2 * w2)
        v_interp = (V0 * w0) + (V1 * w1) + (V2 * w2)
        sampled_texture = torch.clamp(processed_texture * u_interp * v_interp, min=0.0, max=1.0)
        print(f"  [Step 6] Sampled Texture NaN : {torch.isnan(sampled_texture).any().item()}")

        sum_inv_z = torch.conv2d(pixel_inv_z, rast_model.sum_kernel)
        z_diff = torch.relu(sum_inv_z - pixel_inv_z)
        z_blend_weights = torch.clamp(1.0 - (z_diff * 10.0), min=0.0, max=1.0)
        final_mask = mask * z_blend_weights
        print(f"  [Step 7] Final Z Mask HasNaN  : {torch.isnan(final_mask).any().item()}")

        color_payload = sampled_texture * final_mask
        print(f"  [Step 8] Color Payload HasNaN : {torch.isnan(color_payload).any().item()}")

        # 最終Conv集約
        R = torch.conv2d(color_payload, rast_model.sum_kernel)
        G = torch.conv2d(color_payload, rast_model.sum_kernel)
        B = torch.conv2d(color_payload, rast_model.sum_kernel)
        mask_w = torch.conv2d(final_mask, rast_model.sum_kernel)
        max_inv_z = torch.conv2d(pixel_inv_z * z_blend_weights, rast_model.sum_kernel)

        print("==================================================\n")


if __name__ == "__main__":
    main()
