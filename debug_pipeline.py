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

    print("\n==================================================")
    print(" 🔍 FINAL RASTERIZER OUTPUT INFO")
    print("==================================================")
    print(f"  R Channel Max/Min     : {R.max().item()} / {R.min().item()}")
    print(f"  G Channel Max/Min     : {G.max().item()} / {G.min().item()}")
    print(f"  B Channel Max/Min     : {B.max().item()} / {B.min().item()}")
    print(f"  Mask (mask_w) Max/Min : {mask_w.max().item()} / {mask_w.min().item()}")
    print(f"  Max Inv Z Max/Min     : {max_inv_z.max().item()} / {max_inv_z.min().item()}")
    print(f"  Mask HasNaN           : {torch.isnan(mask_w).any().item()}")
    print("==================================================\n")

if __name__ == "__main__":
    main()
