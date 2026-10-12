import coreai_torch
from coreai_torch import TorchConverter
import torch
from RayTracingCore import ANEDynamicSDFCore
from pathlib import Path

# 1. Instance
# 【修正】新しいSDFモデルを使用
model = ANEDynamicSDFCore(max_steps=16).to(dtype=torch.float16)
model.eval()

# 2. Inputs
# Camera Matrix (64チャンネル)
matrix_input_shape = torch.randn(1, 64, 1, 1, dtype=torch.float16)

# 【修正】SDFパラメータ (8図形 × 8パラメータ = 64チャンネル)
sdf_input_shape = torch.randn(1, 128, 1, 1, dtype=torch.float16)

# 【注意】モデルの forward の引数の順番に合わせる
args = (matrix_input_shape, sdf_input_shape)

# 3. Convert
print("Tracing...")
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

# 4. Save
output_path = Path("ane_raytracer.aimodel")
coreai_program.save_asset(output_path)

print(f"Success: `{output_path}`")