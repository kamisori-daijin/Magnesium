import coreai_torch
from coreai_torch import TorchConverter
import torch
from PreProcessor import ANE3DPreProcessor64
from pathlib import Path

WIDTH = 256
HEIGHT = 256

# 1. モデルの初期化 (FP16/Evalモード)
model = ANE3DPreProcessor64().to(dtype=torch.float16)
model.eval()

# -------------------------------------------------------------------------
# 2. ANEに完全最適化した入力ポートの定義 (64をdim=1へ配置)
# -------------------------------------------------------------------------
# 💡 forward(self, expanded_vertices, mvp_weights, colors_r, colors_g, colors_b)

# 頂点バッファ: [バッチ, ポリゴン数(64), 同次座標(4), 頂点数(3)]
dummy_vertices = torch.zeros(1, 64, 4, 3, dtype=torch.float16)

# MVP行列バッファ: [バッチ, ポリゴン数(64), 行列行(4), 行列列(4)]
dummy_mvp_w    = torch.zeros(1, 64, 4, 4, dtype=torch.float16)

# カラーバッファ: 最初から完璧な [1, 64, 1, 1] 形状にする
dummy_r        = torch.zeros(1, 64, 1, 1, dtype=torch.float16)
dummy_g        = torch.zeros(1, 64, 1, 1, dtype=torch.float16)
dummy_b        = torch.zeros(1, 64, 1, 1, dtype=torch.float16)

# 引数の順番を forward の定義と一致させる
args = (dummy_vertices, dummy_mvp_w, dummy_r, dummy_g, dummy_b)

# -------------------------------------------------------------------------
# 3. Core AI向けエクスポート設定
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

coreai_program = converter.to_coreai()
coreai_program.optimize()

# 保存
output_path = Path("ane_3d_pre_processor_64.aimodel")
coreai_program.save_asset(output_path)

print(f"✨ Conversion Success!: `{output_path}`")
