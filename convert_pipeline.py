import coreai_torch
from coreai_torch import TorchConverter
import torch
from pipeline import ANEMonolithicPipeline  
from pathlib import Path

WIDTH = 1024
HEIGHT = 1024

# Initialize Model (Float16)
model = ANEMonolithicPipeline(target_width=WIDTH, target_height=HEIGHT).to(dtype=torch.float16)
model.eval()

# -------------------------------------------------------------------------
# 2. Definition of Input Ports
# -------------------------------------------------------------------------
# Dummy Input Data
expanded_vertices = torch.zeros(1, 64, 4, 3, dtype=torch.float16)
mvp_weights       = torch.zeros(1, 64, 4, 4, dtype=torch.float16)
colors_r          = torch.zeros(1, 64, 1, 1, dtype=torch.float16)
colors_g          = torch.zeros(1, 64, 1, 1, dtype=torch.float16)
colors_b          = torch.zeros(1, 64, 1, 1, dtype=torch.float16)
raw_image         = torch.zeros(1, 3, 256, 256, dtype=torch.float16)

args = (expanded_vertices, mvp_weights, colors_r, colors_g, colors_b, raw_image)

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
output_path = Path("ane_monolithic_pipeline.aimodel")
coreai_program.save_asset(output_path)

print(f"Conversion Success!: `{output_path}`")