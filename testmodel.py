import os
import torch
import torchvision.utils as vutils
from RayTracingCore import ANERayTracingCore


def main():
    print("🎬 ANE特化型：マルチビュー3D削り出しレイトレーシングのテストを開始します...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"-> 使用デバイス: {device}")

    # 1. 外部入力（Input Shape: [1, 3, 256, 256]）のダミー3面図テクスチャを作成
    # ここでは例として「正面は丸(球体)、真上は四角、真横は四角」というマスクを作ります。
    # この3面図を交差させると、空間には自動的に『円柱（シリンダー）』が削り出されます！
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

    # 2. モデルの初期化
    model = ANERayTracingCore().to(device).half()
    model.eval()

    with torch.no_grad():
        print("🚀 [1, 3, 256, 256] の3面図を流し込み、立体を削り出して影を落とします...")
        # 【重要】外部から3面図テクスチャを入力として流し込む！
        output_color = model(dummy_input)

    output_image = output_color.float().cpu()
    output_filename = "ray_trace_multiview.png"
    vutils.save_image(output_image, output_filename, normalize=False)
    print(f"✨ 成功しました！削り出された3Dオブジェクトの画像を保存しました: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    main()