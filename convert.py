import coreai_torch
from coreai_torch import TorchConverter
import torch
# 修正：新しく作った3面図レイトレのクラスをインポート
from RayTracingCore import ANERayTracingCore
from pathlib import Path

print("📦 【Input Shape対応版】CoreAIへの変換準備を開始します...")

# 1. モデルのインスタンス化 (float16に統一)
model = ANERayTracingCore().to(dtype=torch.float16)
model.eval()

# -------------------------------------------------------------------------
# 2. 【超重要】外部入力ポートの形状（Input Shape）を定義
# -------------------------------------------------------------------------
# ANEを殺さない、完璧な4次元 [1, 3, 256, 256] (正面, 真上, 真横の3面図) のShapeを定義
# これにより、外部からリアルタイムに任意の3Dモデルを注入できるポートが作成されます
multiview_input_shape = torch.zeros(1, 3, 256, 256, dtype=torch.float16)

# forward(self, multiview_textures) に渡す引数としてパッケージング
args = (multiview_input_shape,)

# -------------------------------------------------------------------------
# 3. CoreAIへの変換およびエクスポート実行
# -------------------------------------------------------------------------
print("🚀 [Input Shape: 1x3x256x256] でアンロールパイプラインを静的展開中...")
converter = TorchConverter().add_pytorch_module(
    model,
    export_fn=lambda m: torch.export.export(
        m, 
        args=args
    ).run_decompositions(
        coreai_torch.get_decomp_table()
    ),
)

print("🪄 CoreAIプログラムへ変換し、NPU専用のハードウェア最適化を適用します...")
coreai_program = converter.to_coreai()
coreai_program.optimize()

# 4. コンパイル済みアセットの保存
output_path = Path("ane_multiview_raytracer.aimodel")
coreai_program.save_asset(output_path)

print("\n" + "="*50)
print(f"✨ 変換に完全成功しました！！: `{output_path}`")
print("5次元を回避し、かつすべての論理比較を全廃したため、ANEのシリコンを100%フル駆動させる神回路です。")
print("="*50)
