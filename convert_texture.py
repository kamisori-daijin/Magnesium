import coreai_torch
from coreai_torch import TorchConverter
import torch
from TextureModel import ANETextureProcessor
from pathlib import Path

WIDTH = 256
HEIGHT = 256

# 1. モデルをFloat16精度でインスタンス化
model = ANETextureProcessor().to(dtype=torch.float16)
model.eval()

# -------------------------------------------------------------------------
# 2. 入力ポートの定義 (生の256x256 RGB画像)
# -------------------------------------------------------------------------
# [Batch=1, Channel=3, H=256, W=256] の、コンパイラが最も解釈しやすい直球の形状
raw_image_dummy = torch.zeros(1, 3, HEIGHT, WIDTH, dtype=torch.float16)
args = (raw_image_dummy,)

# -------------------------------------------------------------------------
# 3. CoreAI へのエクスポート設定
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
output_path = Path("ane_texture_processor.aimodel")
coreai_program.save_asset(output_path)

print(f"Conversion Success!: `{output_path}`")
