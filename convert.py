import coreai_torch
from coreai_torch import TorchConverter
import torch
from ShaderModel import ANE3DRenderer  # 最新の汎用3Dカラーレンダラーを指定
from pathlib import Path

# 1. 3Dモデルの初期化（くっきり高解像度版の STEPS=128 に設定）
STEPS = 128
WIDTH = 256
HEIGHT = 256

# ANEの大好物である float16（半精度）でモデルをビルド
model = ANE3DRenderer(steps=STEPS, width=WIDTH, height=HEIGHT).to(dtype=torch.float16)
model.eval()

# -------------------------------------------------------------------------
# 2. 僕たちが毎フレーム外部から注入する「動的引数」のトレースシェイプを定義
# -------------------------------------------------------------------------
# ① 動的にカメラを回転させるための 4x4 ビュー行列 [FP16]
sample_camera = torch.eye(4, dtype=torch.float16)

# ② 自由に位置を変えられる3つの球体パラメータ (X, Y, Z, 半径) [FP16]
sample_objects = torch.tensor([
    [ 0.0,  0.1,  2.0, 0.5],  # 1個目: 中央
    [-0.7,  0.2,  2.4, 0.3],  # 2個目: 左奥
    [ 0.6, -0.1,  1.7, 0.35], # 3個目: 右手前
], dtype=torch.float16)

# -------------------------------------------------------------------------
# 3. CoreAI 用のエクスポート設定（引数を2つ追加してトレース）
# -------------------------------------------------------------------------
# 引数に (sample_camera, sample_objects) を指定してエクスポート。
# これによりコンパイラが「外からカメラと物体が動的に流れてくる」ことを理解します。
converter = TorchConverter().add_pytorch_module(
    model,
    export_fn=lambda m: torch.export.export(
        m, 
        args=(sample_camera, sample_objects)
    ).run_decompositions(
        coreai_torch.get_decomp_table()
    ),
)

# -------------------------------------------------------------------------
# 4. CoreAIプログラムのビルド・ANE最適化
# -------------------------------------------------------------------------
coreai_program = converter.to_coreai()

# 1x1 グループConvや次元入れ替え（permute/reshape）が、ANE専用命令へと融合（Fusion）されます
coreai_program.optimize()

# 資産（アセット）として保存
output_path = Path("ane_3d_renderer.aimodel")
coreai_program.save_asset(output_path)

print(f"動的カメラ・物体パラメータ対応のCoreAIモデルが完成しました: `{output_path}`")
print("これでSwiftや別環境から、毎フレーム変化するカメラ行列を一撃注入してグリグリ回転させられます！")
