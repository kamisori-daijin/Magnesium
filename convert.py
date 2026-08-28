import coreai_torch
from coreai_torch import TorchConverter
import torch
from ShaderModel import ANE3DRenderer64
from pathlib import Path

WIDTH = 256
HEIGHT = 256

# 1. モデルの初期化 (FP16/Evalモード)
model = ANE3DRenderer64(width=WIDTH, height=HEIGHT).to(dtype=torch.float16)
model.eval()

# 2. 形状 [1, 64, 1, 1] のダミーバッファ
def make_param():
    return torch.zeros(1, 64, 1, 1, dtype=torch.float16)

# 💡 forward の引数名と完全に一致する辞書型（kwargs）を作成
# これにより、順番の間違いや「引数が見つからない」エラーを 100% 防ぎます
kwargs = {
    # 幾何エッジ (9個)
    "A0": make_param(), "B0": make_param(), "C0": make_param(),
    "A1": make_param(), "B1": make_param(), "C1": make_param(),
    "A2": make_param(), "B2": make_param(), "C2": make_param(),
    
    # 将来カラーブレンドで使う頂点カラー (9個)
    "R0": make_param(), "G0": make_param(), "B0_col": make_param(),
    "R1": make_param(), "G1": make_param(), "B1_col": make_param(),
    "R2": make_param(), "G2": make_param(), "B2_col": make_param(),
    
    # 深度逆数 (3個)
    "p0_iz": make_param(), "p1_iz": make_param(), "p2_iz": make_param(),
    
    # UV座標 (6個)
    "U0": make_param(), "V0": make_param(),
    "U1": make_param(), "V1": make_param(),
    "U2": make_param(), "V2": make_param(),
    
    # テクスチャ (1個)
    "processed_texture": torch.zeros(1, 64, HEIGHT, WIDTH, dtype=torch.float16)
}

# -------------------------------------------------------------------------
# 3. Core AI向けエクスポート設定 (kwargs を渡す形に修正)
# -------------------------------------------------------------------------
converter = TorchConverter().add_pytorch_module(
    model,
    export_fn=lambda m: torch.export.export(
        m, 
        args=(),      # 位置引数は空にする
        kwargs=kwargs # 名前付き引数で安全にマッピング！
    ).run_decompositions(
        coreai_torch.get_decomp_table()
    ),
)

# 4. Core AI プログラムの生成とANE最適化
coreai_program = converter.to_coreai()
coreai_program.optimize()

# 5. 保存
output_path = Path("ane_3d_rasterizer_64.aimodel")
coreai_program.save_asset(output_path)

print(f"Conversion Success!: `{output_path}`")
