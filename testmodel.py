import os
import torch
import numpy as np
import torchvision.utils as vutils
# 新しい仮想RTコアクラスをインポート（ファイル名に合わせて変更してください）
from RayTracingCore import ANEVirtualRTCore

def create_inverse_view_matrix(eye, target, up):
    """
    カメラのビュー逆行列（4x4）を生成
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
    return torch.from_numpy(inv_view).float()

def to_ane_12ch(mat4x4):
    """
    4x4行列から無駄な最下行を排除し、3x4 (12要素) のフラット配列にする
    """
    return mat4x4[:3, :].flatten()

def generate_test_scene_geometry():
    """
    仮想RTコア回路テスト用の静的なシーンジオメトリ（BVH箱とポリゴン頂点）をパック生成。
    今回は中央に配置した1辺0.8の立方体をカバーする箱と、ポリゴンを想定。
    """
    # --- 1. BVH用 AABB（軸平行境界ボックス）データの作成 ---
    # 箱0: 中央のオブジェクトを囲うバウンディングボックス
    box0_min = [-0.5, -0.5, -0.5]
    box0_max = [ 0.5,  0.5,  0.5]
    box0_feat = box0_min + box0_max # 6要素
    
    # 64chの完全アライメントバッファにパッキング
    bvh_boxes_64 = torch.zeros(64, dtype=torch.float32)
    bvh_boxes_64[0:6] = torch.tensor(box0_feat, dtype=torch.float32)
    # 残り（1〜3番目の箱、および空きスペース）は0のままパディング
    
    # --- 2. ポリゴン（三角形面）データの作成 ---
    # 三角形0（手前の面の一部）
    v0 = [-0.4, -0.4,  0.4]
    v1 = [ 0.4, -0.4,  0.4]
    v2 = [-0.4,  0.4,  0.4]
    poly0_feat = v0 + v1 + v2 # 9要素
    
    # 三角形1（手前の面の残り半分）
    v3 = [ 0.4, -0.4,  0.4]
    v4 = [ 0.4,  0.4,  0.4]
    v5 = [-0.4,  0.4,  0.4]
    poly1_feat = v3 + v4 + v5 # 9要素
    
    # 256chの完全アライメントバッファにパッキング
    poly_vertices_256 = torch.zeros(256, dtype=torch.float32)
    poly_vertices_256[0:9] = torch.tensor(poly0_feat, dtype=torch.float32)
    poly_vertices_256[9:18] = torch.tensor(poly1_feat, dtype=torch.float32)
    # 残りのポリゴン枠（2〜15枚目）および空きスペースは0のままパディング
    
    return bvh_boxes_64, poly_vertices_256

def main():
    print("Starting Virtual RT Core testing script...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"-> Using Device: {device}")

    # フレーム保存用ディレクトリ
    os.makedirs("anim_frames", exist_ok=True)

    # 仮想RTコアの初期化
    max_boxes = 4
    max_polygons = 16
    model = ANEVirtualRTCore(max_boxes=max_boxes, max_polygons=max_polygons).to(device).half()
    model.eval()

    # シーンの形状バッファを生成し、デバイスへ転送（形状は固定の実験）
    bvh_boxes_64, poly_vertices_256 = generate_test_scene_geometry()
    bvh_boxes_4d = bvh_boxes_64.view(1, 64, 1, 1).to(device).half()
    poly_vertices_4d = poly_vertices_256.view(1, 256, 1, 1).to(device).half()

    # 30フレームのアニメーションループ
    num_frames = 30
    print(f"Rendering {num_frames} frames: Executing Hardware-Level Circuit Simulation...")

    with torch.no_grad():
        for frame in range(num_frames):
            # 軌道カメラの角度計算
            angle = (frame / num_frames) * 2.0 * np.pi

            # カメラ座標（中央のオブジェクトの周りを旋回）
            cam_x = 2.5 * np.sin(angle)
            cam_y = 0.8 * np.sin(angle * 0.5)
            cam_z = 2.5 * np.cos(angle)
            
            # 4x4のビュー逆行列から、ANE専用の12ch(3x4)フラットデータを抽出
            inv_view_12 = to_ane_12ch(create_inverse_view_matrix(
                eye=[cam_x, cam_y, cam_z], 
                target=[0.0, 0.0, 0.0], 
                up=[0.0, 1.0, 0.0]
            ))

            # ANEが最も喜ぶ「64チャンネル」の入力アライメントバッファへパッキング
            inv_view_64 = torch.zeros(64, dtype=torch.float32)
            inv_view_64[0:12] = inv_view_12 # 0〜11chに3x4行列を格納。12〜63chは無料の空き枠
            
            # 四次元テンソル形状にアライメント変形
            inv_view_4d = inv_view_64.view(1, 64, 1, 1).to(device).half()

            # 仮想回路シミュレーションを実行（引数の並びを新仕様に完全統合）
            # 出力形状: [1, 2, 256, 256] -> (ch0: 最前面深度min_depth, ch1: 描画マスクrender_mask)
            output_buffer = model(inv_view_4d, bvh_boxes_4d, poly_vertices_4d)

            # -----------------------------------------------------------------
            # 画像可視化のためのポスト処理レーン
            # -----------------------------------------------------------------
            # 視覚的に分かりやすいよう、ch1の「衝突マスク(0 or 1)」をRGB画像として出力
            render_mask_2d = output_buffer[:, 1:2, :, :].float().cpu()
            
            # もし深度画像（デプス）として保存したい場合は、以下のようにコメントアウトを切り替えてください
            # depth_2d = output_buffer[:, 0:1, :, :].float().cpu()
            # render_mask_2d = torch.clamp(1.0 - (depth_2d / 5.0), 0.0, 1.0) # 近いほど白くする視覚化

            # 3chのRGB形式に擬似拡張して保存
            output_image = torch.cat([render_mask_2d, render_mask_2d, render_mask_2d], dim=1)
            output_filename = f"anim_frames/frame_{frame:03d}.png"
            
            vutils.save_image(output_image, output_filename, normalize=False)
            print(f" Frame {frame+1}/{num_frames} Circuit Streamed -> {output_filename}")

        print("\n==================================================")
        print("Success! All hardware-emulated frames written to 'anim_frames/'")
        print("==================================================")
    
if __name__ == "__main__":
    main()
