import coreai_torch
from coreai_torch import TorchConverter
import torch
from MVPProcessor import ANEMVPProcessor
from pathlib import Path


MAX_VERTICES = 65536
model = ANEMVPProcessor(max_vertices=MAX_VERTICES).to(dtype=torch.float16)
model.eval()


sample_camera = torch.eye(4, dtype=torch.float16)

sample_vertices = torch.zeros(1, 4, 1, MAX_VERTICES, dtype=torch.float16)


converter = TorchConverter().add_pytorch_module(
    model,
    export_fn=lambda m: torch.export.export(
        m, 
        args=(sample_camera, sample_vertices) 
    ).run_decompositions(
        coreai_torch.get_decomp_table()
    ),
)


coreai_program = converter.to_coreai()


coreai_program.optimize()

# Save
output_path = Path("ane_mvp_processor.aimodel")
coreai_program.save_asset(output_path)

print(f"Conversion Success!: `{output_path}`")

