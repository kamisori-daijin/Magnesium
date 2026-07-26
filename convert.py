import coreai_torch
from coreai_torch import TorchConverter
import torch
from ShaderModel import ANE3DRenderer64  
from pathlib import Path

WIDTH = 256
HEIGHT = 256


model = ANE3DRenderer64(width=WIDTH, height=HEIGHT).to(dtype=torch.float16)
model.eval()

# -------------------------------------------------------------------------
# 2. Definition of Input Ports (64 Triangles Data)
# -------------------------------------------------------------------------
# Create dummy data for coefficients like A0, B0, C0,
def make_dummy():
    return torch.zeros(1, 1, 1, 64, dtype=torch.float16)


# Prepare dummy data matching the arguments of the forward method

args = tuple([make_dummy() for _ in range(19)]) 

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
