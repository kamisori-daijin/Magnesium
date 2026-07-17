import coreai_torch
from coreai_torch import TorchConverter
import torch
from MVPProcessor import ANEMVPProcessor
from pathlib import Path

# 1. 最大65536頂点対応のMVPプロセッサを初期化
# ANEの絶対ルールである float16（半精度）でモデルを構築します
MAX_VERTICES = 65536
model = ANEMVPProcessor(max_vertices=MAX_VERTICES).to(dtype=torch.float16)
model.eval()

# -------------------------------------------------------------------------
# 2. 僕たちが実機（ランタイム）で毎フレーム動的に流し込む「2つの入力ポート」
# -------------------------------------------------------------------------
# ① 【動的カメラ】格闘の末に完成した、ど真ん中に映る4x4のMVP行列 [FP16]
sample_camera = torch.eye(4, dtype=torch.float16)

# ② 【動的頂点バッファ】どんな3Dモデルでも流し込める生のXYZW座標配列 [1, 4, MAX_VERTICES]
# 実機で動かす時は、この固定枠（65536）の中に描きたいモデルの頂点データを詰めて送ります
sample_vertices = torch.zeros(1, 4, MAX_VERTICES, dtype=torch.float16)

# -------------------------------------------------------------------------
# 3. CoreAI 用のエクスポート設定（動的インプットを2つ並べてトレース）
# -------------------------------------------------------------------------
converter = TorchConverter().add_pytorch_module(
    model,
    export_fn=lambda m: torch.export.export(
        m, 
        args=(sample_camera, sample_vertices) # カメラと頂点の動的ポートを同時に開ける！
    ).run_decompositions(
        coreai_torch.get_decomp_table()
    ),
)

# -------------------------------------------------------------------------
# 4. CoreAIプログラムのビルドとANE最適化
# -------------------------------------------------------------------------
coreai_program = converter.to_coreai()

# ここで einsum と割り算がANEネイティブの超高速1x1畳み込み命令に焼き固められます
coreai_program.optimize()

# 資産（アセット）として保存
output_path = Path("ane_mvp_processor.aimodel")
coreai_program.save_asset(output_path)

print(f"🎉 汎用・超高速頂点プロセッサのビルドが完了しました: `{output_path}`")
print("これであらゆる3Dモデルを一瞬で画面の正しい2D座標にプレスする無敵のコアアセットが完成です！")
