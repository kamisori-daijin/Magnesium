import coreai_torch
from coreai_torch import TorchConverter
import torch
from PreProcessor import ANE3DPreProcessor64
from pathlib import Path

WIDTH = 256
HEIGHT = 256

# Cast to float16 
model = ANE3DPreProcessor64().to(dtype=torch.float16)
model.eval()

# -------------------------------------------------------------------------
# 2. Definition of Input Ports 
# -------------------------------------------------------------------------
# forward(self, mvp_matrices, expanded_vertices, colors) 

dummy_vertices = torch.zeros(1, 4, 3, 64, dtype=torch.float16)  # Vertex buffer
dummy_mvp_w    = torch.zeros(1, 4, 4, 1, 64, dtype=torch.float16)  # 1x1 Conv weight shape MVP
dummy_r        = torch.zeros(1, 1, 1, 64, dtype=torch.float16)
dummy_g        = torch.zeros(1, 1, 1, 64, dtype=torch.float16)
dummy_b        = torch.zeros(1, 1, 1, 64, dtype=torch.float16)

args = (dummy_vertices, dummy_mvp_w, dummy_r, dummy_g, dummy_b)

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
output_path = Path("ane_3d_pre_processor_64.aimodel")
coreai_program.save_asset(output_path)

print(f"Conversion Success!: `{output_path}`")
