import coreai_torch
from coreai_torch import TorchConverter
import torch
from ShaderModel import ANE3DRenderer64  # 64枚版のラスタライザ
from pathlib import Path

WIDTH = 256
HEIGHT = 256

# ★【ANE最適化の鉄則】
# 一度にすべてを処理するのではなく、ANEが最も得意とする「64枚」のチャンクに分割して変換します。
model = ANE3DRenderer64(width=WIDTH, height=HEIGHT).to(dtype=torch.float16)
model.eval()

# -------------------------------------------------------------------------
# 2. 入力ポートの定義（64枚の三角形データ）
# -------------------------------------------------------------------------
# A0, B0, C0 などの係数と、RGBカラー、Zウェイトのダミーデータを作成
def make_dummy():
    return torch.zeros(1, 1, 1, 64, dtype=torch.float16)

# forwardメソッドの引数に合わせてダミーデータを用意
args = tuple([make_dummy() for _ in range(19)]) # 19個の引数

# -------------------------------------------------------------------------
# 3. CoreAI 用のエクスポート設定
# -------------------------------------------------------------------------
converter = TorchConverter().add_pytorch_module(
    model,
    export_fn=lambda m: torch.export.export(
        m, 
        args=args
    ).run_decompositions(
        coreai_torch.get_decomp_table()
    ),
)

# 4. CoreAIプログラムのビルド・ANE最適化
coreai_program = converter.to_coreai()
coreai_program.optimize()

# 資産（アセット）として保存
output_path = Path("ane_3d_rasterizer_64.aimodel")
coreai_program.save_asset(output_path)

print(f"🎉 ANE最適化済みラスタライザのビルドが完了しました: `{output_path}`")
print("64枚単位で処理を行うため、ANEのメモリを溢れさせることなく安全に動作します！")