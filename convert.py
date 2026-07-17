import coreai_torch
from coreai_torch import TorchConverter
import torch
from ShaderModel import ANE3DRenderer  # グリッド地面＆調光内蔵の最新ラスタライザ
from pathlib import Path

# 解像度はテストコードと完全に一致する 256x256、半精度(FP16)
WIDTH = 256
HEIGHT = 256

# ★【ここが最重要ハック】
# 第1段（MVP）の最大頂点数 65536 にポートの幅を100%カチッと一致させます！
# 三角形1枚につき3頂点が必要なため、一度に処理できる最大三角形数は 65536 // 3 = 21845枚 に自動拡張されます。
MAX_VERTICES = 65536
MAX_TRIANGLES = MAX_VERTICES // 3

model = ANE3DRenderer(
    max_triangles=MAX_TRIANGLES, 
    width=WIDTH, 
    height=HEIGHT
).to(dtype=torch.float16)
model.eval()

# -------------------------------------------------------------------------
# 2. 第1段（MVPプロセッサ）の出力形状 [1, 3, 1, 65536] と完全に一致するポートを開ける
# -------------------------------------------------------------------------
sample_transformed_vertices = torch.zeros(1, 3, 1, MAX_VERTICES, dtype=torch.float16)

# -------------------------------------------------------------------------
# 3. CoreAI 用のエクスポート設定
# -------------------------------------------------------------------------
converter = TorchConverter().add_pytorch_module(
    model,
    export_fn=lambda m: torch.export.export(
        m, 
        args=(sample_transformed_vertices,) # 第1段の出力ポート [1, 3, 1, 65536] と完全合体！
    ).run_decompositions(
        coreai_torch.get_decomp_table()
    ),
)

# 4. CoreAIプログラムのビルド・ANE最適化
coreai_program = converter.to_coreai()
coreai_program.optimize()

# 資産（アセット）として保存
output_path = Path("ane_3d_rasterizer.aimodel")
coreai_program.save_asset(output_path)

print(f"🎉 プラグサイズ完全互換・動的ラスタライザのビルドが完了しました: `{output_path}`")
print("これで第1段 [1, 3, 1, 65536] のバトンをポインタのまま100%直撃させられるアセットの完成です！")
