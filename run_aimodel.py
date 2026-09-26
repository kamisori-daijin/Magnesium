import asyncio
from pathlib import Path
import numpy as np
from PIL import Image

from coreai.authoring import AIModelAsset
from coreai.runtime import InferenceFunction, NDArray

# -------------------------------------------------------------------------
# 💡 [宇宙創成ハック] 2つの巨大な回転銀河が激突する初期データを生成する関数
# -------------------------------------------------------------------------
def create_colliding_galaxies(num_channels=128, size=128):
    total_stars = num_channels * size * size  # 128^3 = 2,097,152個
    
    # 210万個の位置と速度を NumPy 配列で初期化
    pos_x = np.zeros((1, num_channels, size, size), dtype=np.float32)
    pos_y = np.zeros((1, num_channels, size, size), dtype=np.float32)
    pos_z = np.zeros((1, num_channels, size, size), dtype=np.float32)
    
    vel_x = np.zeros((1, num_channels, size, size), dtype=np.float32)
    vel_y = np.zeros((1, num_channels, size, size), dtype=np.float32)
    vel_z = np.zeros((1, num_channels, size, size), dtype=np.float32)
    
    # すべての星の質量（均一に 1.0 に設定）
    mass = np.ones((1, num_channels, size, size), dtype=np.float32) * 1.0

    # 210万個をフラットに扱って初期位置と回転速度を計算
    flat_px = pos_x.ravel()
    flat_py = pos_y.ravel()
    flat_pz = pos_z.ravel()
    flat_vx = vel_x.ravel()
    flat_vy = vel_y.ravel()
    flat_vz = vel_z.ravel()

    # 銀河1（中心位置 [-0.5, 0.0, 0.0]）と銀河2（中心位置 [0.5, 0.0, 0.0]）に半分ずつ分ける
    half = total_stars // 2

    # --- 銀河1の生成（左側） ---
    r1 = np.random.rand(half) * 0.4 + 0.05  # 半径
    theta1 = np.random.rand(half) * 2.0 * np.pi
    flat_px[:half] = -0.5 + r1 * np.cos(theta1)
    flat_py[:half] = r1 * np.sin(theta1)
    flat_pz[:half] = (np.random.randn(half) * 0.02)  # 薄い円盤にするためZ軸は極小のブレ
    
    # 回転初速度 (ケプラー回転風に引力と遠心力をバランスさせて渦を巻かせる)
    v_mag1 = np.sqrt(0.0001 / (r1 + 1e-3))
    flat_vx[:half] = -v_mag1 * np.sin(theta1) + 0.05  # 右（相手方向）への移動速度を足す
    flat_vy[:half] = v_mag1 * np.cos(theta1)
    flat_vz[:half] = 0.0

    # --- 銀河2の生成（右側） ---
    r2 = np.random.rand(half) * 0.4 + 0.05
    theta2 = np.random.rand(half) * 2.0 * np.pi
    flat_px[half:] = 0.5 + r2 * np.cos(theta2)
    flat_py[half:] = r2 * np.sin(theta2)
    flat_pz[half:] = (np.random.randn(half) * 0.02)
    
    v_mag2 = np.sqrt(0.0001 / (r2 + 1e-3))
    flat_vx[half:] = -v_mag2 * np.sin(theta2) - 0.05  # 左（相手方向）への移動速度を足す
    flat_vy[half:] = v_mag2 * np.cos(theta2)
    flat_vz[half:] = 0.0

    # ANEが最も愛する Float16 (Half精度) に一斉キャストしてリターン
    return (pos_x.astype(np.float16), pos_y.astype(np.float16), pos_z.astype(np.float16),
            vel_x.astype(np.float16), vel_y.astype(np.float16), vel_z.astype(np.float16),
            mass.astype(np.float16))

async def main():
    # 決定版の引力モデルアセットのパス
    engine_path = Path("./ane_gravity_engine.aimodel")
    
    if not engine_path.exists():
        print(f"Error: Asset `{engine_path}` not found. Please compile it first.")
        return

    print("Loading 2.1 Million Particles Gravity Engine onto ANE...")
    engine_asset = AIModelAsset.load(engine_path)
    
    # 初期宇宙データをNumPyで爆速生成
    print("Creating initial positions and velocities for 2 Colliding Galaxies...")
    px, py, pz, vx, vy, vz, mass = create_colliding_galaxies()
    
    # dtも完全横並び形状で初期化
    dt = np.full((1, 128, 128, 128), 0.02, dtype=np.float16)

    async with engine_asset.executable() as engine_model:
        # CoreAIの推論関数を取得
        gravity_function: InferenceFunction = engine_model.load_function("main")

        print("\n=== Start 4.4 Trillion Interactions ANE Loop ===")
        print("Running 10 frames of cosmic evolution test...")
        
        # 💡 モデルの内部で形状が [1, 128, 128, 128] に復元されて返ってくるため、
        # 2周目以降も形状不一致のエラーを起こさず安全に回り続けます
        for frame in range(1, 11):
            inputs = {
                "all_pos_x": NDArray(px),
                "all_pos_y": NDArray(py),
                "all_pos_z": NDArray(pz),
                "all_vel_x": NDArray(vx),
                "all_vel_y": NDArray(vy),
                "all_vel_z": NDArray(vz),
                "all_mass": NDArray(mass),
                "dt": NDArray(dt)
            }
            
            # ANEをフル稼働させて、一撃で次の宇宙のステートを計算
            outputs = await gravity_function(inputs)
            
            # 各出力ポートから、3本セパレートのまま次のフレームのデータを引き出す
            out_names = gravity_function.desc.output_names
            px = outputs[out_names[0]].numpy()
            py = outputs[out_names[1]].numpy()
            pz = outputs[out_names[2]].numpy()
            vx = outputs[out_names[3]].numpy()
            vy = outputs[out_names[4]].numpy()
            vz = outputs[out_names[5]].numpy()

            # 宇宙の中心付近の星の位置をトラッキング
            print(f" Frame [{frame:02d}] ANE Physics Done. Star 0 Position -> X: {px[0, 0, 0, 0]:.4f}, Y: {py[0, 0, 0, 0]:.4f}, Z: {pz[0, 0, 0, 0]:.4f}")

        print("\n✨ Simulation Loop finished successfully on Apple Neural Engine!")
        print(f"Final shape: {px.shape} (Perfect 4D Tensor format)")

        # -------------------------------------------------------------------------
        # 📸 最終フレームの210万天体データを2D平面にプロットして画像保存
        # -------------------------------------------------------------------------
        print("\n📸 Rendering 2.1 Million Particles into PNG image...")
        
        # 256x256 の真っ黒なキャンバスを用意
        canvas = np.zeros((256, 256), dtype=np.float32)
        
        # モデル内部の slice_update で書き戻した、有効な左上65x65領域の星屑（計528,125個）を抽出
        stars_x = px[0, :, :65, :65].ravel()
        stars_y = py[0, :, :65, :65].ravel()
        
        # 宇宙空間の座標（-1.0 〜 1.0）を、画像平面のピクセルインデックス（0 〜 255）に一斉変換
        screen_x = ((stars_x + 1.0) * 127.5).astype(np.int32)
        screen_y = ((stars_y + 1.0) * 127.5).astype(np.int32)
        
        # 画面の範囲内（0〜255）に収まっている星だけを抽出
        valid_indices = (screen_x >= 0) & (screen_x < 256) & (screen_y >= 0) & (screen_y < 256)
        plot_x = screen_x[valid_indices]
        plot_y = screen_y[valid_indices]
        
        # 同じピクセルに星が密集するほど輝度が累積（np.add.at で超並列加算）
        np.add.at(canvas, (plot_y, plot_x), 0.1)
        
        # 輝度を 0.0 〜 1.0 にクランプして、8bitの白黒画像（0〜255）に変換
        final_img_data = (np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8)
        
        # PNGとしてカレントディレクトリに保存
        output_img_path = "ane_galaxy_collision_final.png"
        Image.fromarray(final_img_data, 'L').save(output_img_path)
        
        print(f"✨ Cosmic Snapshot saved successfully!: `{output_img_path}`")
        print(f"Total stars plotted on screen: {len(plot_x)} / {len(stars_x)}")

if __name__ == "__main__":
    asyncio.run(main())
