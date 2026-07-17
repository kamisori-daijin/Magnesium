import coreai_torch
from coreai_torch import TorchConverter
import torch
from ShaderModel import ANE3DRenderer  
from pathlib import Path

# 1. モデルの初期化（128ステップ）
STEPS = 128
WIDTH = 256
HEIGHT = 256
model = ANE3DRenderer(steps=STEPS, width=WIDTH, height=HEIGHT).to(dtype=torch.float16)
model.eval()

# -------------------------------------------------------------------------
# 2. 僕たちが毎フレーム動的に流し込む「2つの入力ポート」のシェイプ定義
# -------------------------------------------------------------------------
# ① 外部カメラ行列 [4, 4]
sample_camera = torch.eye(4, dtype=torch.float16)

# ② 自由に配置・移動できる3つのオブジェクト枠 [3, 4] (X, Y, Z, Radius)
# コンパイル時にこの「オブジェクト最大数（3個）」を固定枠として焼き付けます
sample_objects = torch.tensor([
    [ 0.0,  0.1,  2.0, 0.5], 
    [-0.7,  0.2,  2.4, 0.3], 
    [ 0.6, -0.1,  1.7, 0.35],
], dtype=torch.float16)

# -------------------------------------------------------------------------
# 3. CoreAI 用のエクスポート設定（動的インプットを2つ並べてトレース）
# -------------------------------------------------------------------------
converter = TorchConverter().add_pytorch_module(
    model,
    export_fn=lambda m: torch.export.export(
        m, 
        args=(sample_camera, sample_objects) # 2つの動的ポートを同時に開ける！
    ).run_decompositions(
        coreai_torch.get_decomp_table()
    ),
)

# 4. CoreAIプログラムのビルドと最適化
coreai_program = converter.to_coreai()
coreai_program.optimize()

# 保存
output_path = Path("ane_3d_renderer_universal.aimodel")
coreai_program.save_asset(output_path)

print(f"動的カメラ＆動的複数オブジェクト対応のユニバーサル3Dモデルが完成しました: `{output_path}`")
