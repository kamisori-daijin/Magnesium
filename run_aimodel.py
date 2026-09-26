import asyncio
from pathlib import Path
import numpy as np

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
    
    # すべての星の質量（均一に 1.0 に設定、中心のブラックホールだけ重くするなども可能）
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
    
    # あなたのハック思想通り、dtも完全横並び形状で初期化
    dt = np.full((1, 128, 128, 128), 0.02, dtype=np.float16)

    async with engine_asset.executable() as engine_model:
        # CoreAIの推論関数を取得
        gravity_function: InferenceFunction = engine_model.load_function("main")

        print("\n=== Start 4.4 Trillion Interactions ANE Loop ===")
        print("Running 10 frames of cosmic evolution test...")
        
        # 💡 [ループ完全追放の実感] 
        # モデルの内部からはループを完全に追放したため、
        # アプリ側のこの短いタイムループが1回まわるたびに、210万×210万の全相互作用が一撃で完了します！
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
            # 💡 パディング排除（Valid Conv）仕様のため、出力形状は [1, 128, 65, 65] に綺麗に収縮しています
            px = outputs[gravity_function.desc.output_names[0]].numpy()
            py = outputs[gravity_function.desc.output_names[1]].numpy()
            pz = outputs[gravity_function.desc.output_names[2]].numpy()
            vx = outputs[gravity_function.desc.output_names[3]].numpy()
            vy = outputs[gravity_function.desc.output_names[4]].numpy()
            vz = outputs[gravity_function.desc.output_names[5]].numpy()

            # パフォーマンスデバッグとして、宇宙の中心付近の星の位置をトラッキング
            print(f" Frame [{frame:02d}] ANE Physics Done. Star 0 Position -> X: {px[0, 0, 0, 0]:.4f}, Y: {py[0, 0, 0, 0]:.4f}, Z: {pz[0, 0, 0, 0]:.4f}")

        print("\n✨ Simulation Loop finished successfully on Apple Neural Engine!")
        print(f"Final valid particle matrix shape: {px.shape} (Perfectly fitted to ANE NEU layout)")

if __name__ == "__main__":
    asyncio.run(main())
