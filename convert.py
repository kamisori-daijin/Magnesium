import coreai_torch
from coreai_torch import TorchConverter
import torch
from ShaderModel import ANE3DRenderer64
from pathlib import Path

WIDTH = 256
HEIGHT = 256

# モデルをfloat16で初期化
model = ANE3DRenderer64(width=WIDTH, height=HEIGHT).to(dtype=torch.float16)
model.eval()

# -------------------------------------------------------------------------
# 2. 入力ポートの定義 (最適化版のShapeに合わせる)
# -------------------------------------------------------------------------
# エッジ係数 (weights_0, 1, 2) 用のダミー：[64, 3, 1, 1] に変更
def make_edge_dummy():
    return torch.zeros(64, 3, 1, 1, dtype=torch.float16)

# その他のパラメータ用（p0_iz, U0, V0など）のダミー：[1, 64, 1, 1] 
def make_param_dummy():
    return torch.zeros(1, 64, 1, 1, dtype=torch.float16)

# forwardの引数の順番に合わせてタプルを構築
args = (
    make_edge_dummy(),  # weights_0
    make_edge_dummy(),  # weights_1
    make_edge_dummy(),  # weights_2
    
    make_param_dummy(), # p0_iz
    make_param_dummy(), # p1_iz
    make_param_dummy(), # p2_iz
    
    make_param_dummy(), # U0
    make_param_dummy(), # V0
    make_param_dummy(), # U1
    make_param_dummy(), # V1
    make_param_dummy(), # U2
    make_param_dummy(), # V2
    
    torch.zeros(1, 64, 256, 256, dtype=torch.float16) # processed_texture
)

# -------------------------------------------------------------------------
# 3. Core AI向けエクスポート設定
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

# 保存
output_path = Path("ane_3d_rasterizer_64_optimized.aimodel")
coreai_program.save_asset(output_path)

print(f"Conversion Success!: `{output_path}`")
