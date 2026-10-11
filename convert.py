import coreai_torch
from coreai_torch import TorchConverter
import torch
from RayTracingCore import ANEDynamicRayTracingCore
from pathlib import Path

# 1. Instance
grid_size = 8
model = ANEDynamicRayTracingCore(grid_size=grid_size).to(dtype=torch.float16)
model.eval()

# 2. Inputs
# ボクセルデータ: grid_size^3 の要素数を持つ4次元テンソル
num_voxels = grid_size ** 3
voxel_input_shape = torch.zeros(1, num_voxels, 1, 1, dtype=torch.float16)

# Camera Matrix
matrix_input_shape = torch.zeros(1, 64, 1, 1, dtype=torch.float16)

args = (voxel_input_shape, matrix_input_shape)

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