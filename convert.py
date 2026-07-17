import coreai_torch
from coreai_torch import TorchConverter
import torch
from ShaderModel import ANE3DRenderer  
from pathlib import Path


STEPS = 64
WIDTH = 256
HEIGHT = 256
model = ANE3DRenderer(steps=STEPS, width=WIDTH, height=HEIGHT).to(dtype=torch.float16)
model.eval()

# -------------------------------------------------------------------------
# 2. カメラ行列の初期化（ダミーのウェイトとしてあらかじめセットしておく）
# -------------------------------------------------------------------------
# ANEのConv2dのウェイト形状 [4, 4, 1, 1] に適合する4x4の単位行列を作ってセット
init_camera = torch.eye(4, dtype=torch.float16)
model.update_camera(init_camera)

# -------------------------------------------------------------------------
# 3. CoreAI 用のエクスポート設定（トレースシェイプの変更と追加）
# -------------------------------------------------------------------------
# 新しい forward は引数を取らないため、args は空のタプル `()` に変更します。
# 内部の static_space や内部バッファがすべて fp16 でトレースされます。
converter = TorchConverter().add_pytorch_module(
    model,
    export_fn=lambda m: torch.export.export(m, args=()).run_decompositions(
        coreai_torch.get_decomp_table()
    ),
)

# -------------------------------------------------------------------------
# 4. CoreAIプログラムのビルドと最適化
# -------------------------------------------------------------------------
coreai_program = converter.to_coreai()

# ANE専用のレイアウト最適化や1x1 Convの融合（Fusion）がここで走ります
coreai_program.optimize()

# 資産（アセット）として保存
output_path = Path("ane_3d_renderer.aimodel")
coreai_program.save_asset(output_path)

print(f"CoreAIモデルのビルドが完了しました: `{output_path}`")
print("カメラ行列を動的に外部から注入したい場合は、さらに一工夫必要ですが、まずはこの形状でANEに一撃プレス可能です！")
