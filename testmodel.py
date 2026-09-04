import os
import torch
import torchvision.utils as vutils
from RayTracingCore import ANERayTracingCore


# 動作確認用の実行スクリプト
def main():
    print("🎬 ANE特化型レイトレーシング（レイマーチング）のテストを開始します...")
    
    # 1. デバイスの設定 (手元のMacなら 'mps'、Windows/Linuxなら 'cuda'、それ以外は 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("-> Apple Silicon (MPS) 加速を使用します。")
    else:
        device = torch.device("cpu")
        print("-> CPUを使用します。")

    # 2. モデルの初期化（解像度 256x256、前進ステップ数 24回に増やして精度向上）
    # ANEの仕様に合わせて float16 にキャストします
    model = ANERayTracingCore(width=256, height=256, max_steps=24).to(device).half()
    model.eval()  # 推論モード

    # 3. レイトレーシング（バコン！と1発実行）
    with torch.no_grad():
        print("🚀 レイを256x256本一斉に放ちます（Whereなし算術ブレンドスキャン中）...")
        # 出力: [1, 1, 256, 256] の不透明度マスク (0.0=背景, 1.0=球体にヒット)
        output_mask = model()

    # 4. 画像データとして保存するために float32 に戻してCPUへ転送
    output_image = output_mask.float().cpu()

    # 5. 画像として保存 (torchvision を使用)
    output_filename = "ray_trace_output.png"
    vutils.save_image(output_image, output_filename, normalize=True)
    
    print(f"✨ 完了しました！出力結果を画像として保存しました: {os.path.abspath(output_filename)}")
    print(f"画像サイズ: {output_image.shape[2]}x{output_image.shape[3]} (ピクセル)")
    print(f"最大値: {output_image.max().item():.1f}, 最小値: {output_image.min().item():.1f}")

if __name__ == "__main__":
    main()
