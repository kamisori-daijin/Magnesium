import coreai_torch
from coreai_torch import TorchConverter
import torch
from ShaderModel import ANEGravityEngine  # あなたが作成した決定版モデル
from pathlib import Path

# -------------------------------------------------------------------------
# 1. モデルのインスタンス化 (ANE最適化のためにFloat16で統一)
# -------------------------------------------------------------------------
model = ANEGravityEngine().to(dtype=torch.float16)
model.eval()

# -------------------------------------------------------------------------
# 2. ANEが最も喜ぶ「1, 128, 128, 128」の完全均一4次元形状の定義
# -------------------------------------------------------------------------
# 210万天体の位置（X, Y, Z）の3本
pos_x_shape = torch.zeros(1, 128, 128, 128, dtype=torch.float16)
pos_y_shape = torch.zeros(1, 128, 128, 128, dtype=torch.float16)
pos_z_shape = torch.zeros(1, 128, 128, 128, dtype=torch.float16)

# 210万天体の速度（Vx, Vy, Vz）の3本
vel_x_shape = torch.zeros(1, 128, 128, 128, dtype=torch.float16)
vel_y_shape = torch.zeros(1, 128, 128, 128, dtype=torch.float16)
vel_z_shape = torch.zeros(1, 128, 128, 128, dtype=torch.float16)

# 各天体の質量（Mass）の1本
mass_shape  = torch.zeros(1, 128, 128, 128, dtype=torch.float16)

# 💡 ANEハック: あなたの閃き通り、dtも他のポートと1ミリの狂いもない同一形状で流し込む！
dt_shape    = torch.full((1, 128, 128, 128), 0.01, dtype=torch.float16)

# forward(all_pos_x, all_pos_y, all_pos_z, all_vel_x, all_vel_y, all_vel_z, all_mass, dt) の順に格納
args = (
    pos_x_shape, pos_y_shape, pos_z_shape, 
    vel_x_shape, vel_y_shape, vel_z_shape, 
    mass_shape, dt_shape
)

# -------------------------------------------------------------------------
# 3. CoreAI（Apple Neural Engine向け）エクスポート設定
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# 4. アセットの保存
# -------------------------------------------------------------------------
output_path = Path("ane_gravity_engine.aimodel")
coreai_program.save_asset(output_path)

print(f"Conversion Success!: `{output_path}`")
