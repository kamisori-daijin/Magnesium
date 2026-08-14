import coreai_torch
from coreai_torch import TorchConverter
import torch
from ShaderModel import ANE3DRenderer64  
from pathlib import Path

# 128x128 
WIDTH = 128
HEIGHT = 128

model = ANE3DRenderer64(width=WIDTH, height=HEIGHT).to(dtype=torch.float16)
model.eval()

# -------------------------------------------------------------------------
# 2. Definition of Input Ports (64 Triangles Data)
# -------------------------------------------------------------------------
def make_dummy():
    return torch.zeros(1, 1, 1, 64, dtype=torch.float16)


args_list = [make_dummy() for _ in range(27)]

# Texture
args_list.append(torch.zeros(1, 64, HEIGHT, WIDTH, dtype=torch.float16))

# tile_offset_x, tile_offset_y
args_list.append(torch.zeros(1, dtype=torch.float16)) # tile_offset_x
args_list.append(torch.zeros(1, dtype=torch.float16)) # tile_offset_y

args = tuple(args_list)

# -------------------------------------------------------------------------
# 3. Export Settings for CoreAI
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

# save
output_path = Path("ane_3d_rasterizer_64.aimodel")
coreai_program.save_asset(output_path)

print(f"Conversion Success!: `{output_path}`")