import coreai_torch
from coreai_torch import TorchConverter
import torch
from RayTracingCore import ANERayTracingCore
from pathlib import Path

print("📦 【Input Shape対応版】CoreAIへの変換準備を開始します...")

# 1. Instance
model = ANERayTracingCore().to(dtype=torch.float16)
model.eval()


multiview_input_shape = torch.zeros(1, 3, 256, 256, dtype=torch.float16)

# Camera
matrix_input_shape = torch.zeros(1, 64, 1, 1, dtype=torch.float16)

args = (multiview_input_shape, matrix_input_shape)


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
coreai_program.optimize()

# 4. Save
output_path = Path("ane_raytracer.aimodel")
coreai_program.save_asset(output_path)


print(f"Success: `{output_path}`")

