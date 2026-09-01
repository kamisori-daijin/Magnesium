import coreai_torch
from coreai_torch import TorchConverter
import torch
from TextureModel import ANETextureProcessor
from pathlib import Path

WIDTH = 1024
HEIGHT = 1024

# 1. Float16
model = ANETextureProcessor().to(dtype=torch.float16)
model.eval()



# [Batch=1, Channel=3, H=256, W=256]
raw_image_dummy = torch.zeros(1, 3, HEIGHT, WIDTH, dtype=torch.float16)
args = (raw_image_dummy,)

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

# Save
output_path = Path("ane_texture_processor.aimodel")
coreai_program.save_asset(output_path)

print(f"Conversion Success!: `{output_path}`")
