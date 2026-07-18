import torch
from PIL import Image
import numpy as np
from MVPProcessor import ANEMVPProcessor
from ShaderModel import ANE3DRenderer64


def run_full_pipeline():
    # 1. モデルの初期化
    mvp_processor = ANEMVPProcessor(max_vertices=3)
    renderer = ANE3DRenderer64(width=256, height=256)
    
    # 2. 3D頂点データ [1, 4, 1, 3] (X, Y, Z, W)
    # 少し奥(Z=2.0)に配置した三角形
    vertices = torch.tensor([
        [ 0.0,  0.8, 2.0, 1.0],
        [-0.8, -0.5, 2.0, 1.0],
        [ 0.8, -0.5, 2.0, 1.0]
    ]).T.unsqueeze(0).unsqueeze(2) # [1, 4, 1, 3] に変形
    
    # 3. カメラ行列 (今回はシンプルな単位行列)
    camera_matrix = torch.eye(4)
    
    # 4. MVPプロセッサで2D座標に変換
    with torch.no_grad():
        transformed = mvp_processor(camera_matrix, vertices)
        
    # 5. 変換された座標を取り出す
    p0_2d = transformed[0, :2, 0, 0].squeeze()
    p1_2d = transformed[0, :2, 0, 1].squeeze()
    p2_2d = transformed[0, :2, 0, 2].squeeze()
    
    # 6. エッジ関数 (A, B, C) の計算
    def get_edge(p_a, p_b):
        A = p_a[1] - p_b[1]
        B = p_b[0] - p_a[0]
        C = -(A * p_a[0] + B * p_a[1])
        return A, B, C

    A0, B0, C0 = get_edge(p0_2d, p1_2d)
    A1, B1, C1 = get_edge(p1_2d, p2_2d)
    A2, B2, C2 = get_edge(p2_2d, p0_2d)

    # 7. 64個のバッチに詰め替え
    def pack(val):
        t = torch.zeros(1, 1, 1, 64)
        t[0, 0, 0, 0] = val
        return t

    # 8. ラスタライザで描画
    with torch.no_grad():
        R, G, B, mask = renderer(
            pack(A0), pack(B0), pack(C0),
            pack(A1), pack(B1), pack(C1),
            pack(A2), pack(B2), pack(C2),
            pack(1.0), pack(0.0), pack(0.0), # 赤
            pack(0.0), pack(1.0), pack(0.0), # 緑
            pack(0.0), pack(0.0), pack(1.0), # 青
            pack(1.0 / transformed[0, 2, 0, 0]) # Zウェイト
        )

    # 9. 画像保存
    r_img, g_img, b_img = R[0, 0].numpy(), G[0, 0].numpy(), B[0, 0].numpy()
    img_array = np.stack([r_img, g_img, b_img], axis=-1)
    img_array = (np.clip(img_array, 0.0, 1.0) * 255).astype(np.uint8)
    
    Image.fromarray(img_array).save("output_pipeline.png")
    print("output_pipeline.png を保存しました！")

if __name__ == "__main__":
    run_full_pipeline()