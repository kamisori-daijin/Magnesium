import coreai_torch
from coreai_torch import TorchConverter
import torch
from ShaderModel import ANEShaderTriangle
from pathlib import Path

model = ANEShaderTriangle().to(dtype=torch.float16)
model.eval()


sample_x = torch.randn(1, 2, 1024, 1024, dtype=torch.float16)
sample_weight = torch.randn(3, 2, 1, 1, dtype=torch.float16)
sample_bias = torch.randn(3, dtype=torch.float16)


converter = TorchConverter().add_pytorch_module(
    model,
    export_fn=lambda m: torch.export.export(m, args=(sample_x, sample_weight, sample_bias)).run_decompositions(
        coreai_torch.get_decomp_table()
    ),
)

coreai_program = converter.to_coreai()
coreai_program.optimize()
coreai_program.save_asset(Path("shader_triangle.aimodel"))


