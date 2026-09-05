import os
import torch
import torchvision.utils as vutils
# クラス名はあなたが提示した元の名前に合わせています
from RayTracingCore import ANERayTracingCore


def main():
    print("🎬 ANE特化型（内部バッチ処理版）：マルチビュー3D削り出しレイトレーシングのテストを開始します...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"-> 使用デバイス: {device}")

    # 1. モデルの初期化（ステップ数は内部で処理されるため変更なし）
    max_steps = 64
    shadow_steps = 16
    model = ANERayTracingCore(max_steps=max_steps, shadow_steps=shadow_steps).to(device).half()
    model.eval()

    # 2. 外部入力（ベース形状）のダミー3面図テクスチャを作成 [1, 3, 256, 256]
    dummy_input = torch.zeros(1, 3, 256, 256, dtype=torch.float16, device=device)
    
    # チャンネル0: 正面図に白い円（丸）を描画
    y, x = torch.meshgrid(torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256), indexing="ij")
    circle_mask = (x*x + y*y) < 0.35
    dummy_input[0, 0, :, :] = circle_mask.to(device).half()
    
    # チャンネル1: 真上図に四角（ボックス）を描画
    box_mask_top = (torch.abs(x) < 0.6) * (torch.abs(y) < 0.6)
    dummy_input[0, 1, :, :] = box_mask_top.to(device).half()
    
    # チャンネル2: 真横図にも四角を描画
    box_mask_side = (torch.abs(x) < 0.6) * (torch.abs(y) < 0.6)
    dummy_input[0, 2, :, :] = box_mask_side.to(device).half()

    # 3. 【神修正】無駄な expand を全廃し、[1, 3, 256, 256] のまま流し込む
    with torch.no_grad():
        print("🚀 [1, 3, 256, 256] の最小テクスチャをそのまま流し込み、立体を削り出します...")
        output_color = model(dummy_input)

    output_image = output_color.float().cpu()
    output_filename = "ray_trace_multiview_batch.png"
    vutils.save_image(output_image, output_filename, normalize=False)
    print(f"✨ 成功しました！ループなしで削り出された3Dオブジェクトの画像を保存しました: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    main()
