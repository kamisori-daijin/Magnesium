import asyncio
from pathlib import Path
from coreai.authoring import AIModelAsset
from coreai.runtime import InferenceFunction, SpecializationOptions

async def inspect_model(model_path: str, label: str):
    print(f"\n==================================================")
    print(f" ⚙️ [ANALYSIS] {label} ('{model_path}')")
    print(f"==================================================")
    
    # 💡 ノートブック準拠: AIModelAsset.load を使用
    path = Path(model_path)
    if not path.exists():
        print(f"Error: {model_path} does not exist.")
        return
        
    asset = AIModelAsset.load(path)
    
    # ノートブック準拠の正式な初期化
    async with asset.executable() as model:
        print(f"Exposed Functions: {model.function_names}")
        
        # main 関数のディスクリプタをロード
        function = model.load_function("main")
        desc = function.desc
        
        print(f"\n📥 INPUT NAMES (Swiftから渡すべき引数名):")
        for inp in desc.input_names:
            print(f"  - '{inp}'")
            
        print(f"\n📤 OUTPUT NAMES (★Swiftの引数に書くべき本物の自動生成名):")
        for idx, outp in enumerate(desc.output_names):
            print(f"  - Output [{idx}]: '{outp}'")

async def main():
    # 1. プリプロセッサの解析
    try:
        await inspect_model("ane_3d_pre_processor_64.aimodel", "3D PreProcessor")
    except Exception as e:
        # 万が一リネーム前のファイル名だった場合のためのフォールバック
        try:
            await inspect_model("ane_3d_pre_processor_64.aimodel", "3D PreProcessor")
        except Exception as e2:
            print(f"PreProcessor Analysis Failed: {e2}")
        
    # 2. ラスタライザの解析
    try:
        await inspect_model("ane_3d_rasterizer_64.aimodel", "3D Rasterizer")
    except Exception as e:
        try:
            await inspect_model("ane_3d_rasterizer_64.aimodel", "3D Rasterizer")
        except Exception as e2:
            print(f"Rasterizer Analysis Failed: {e2}")

if __name__ == "__main__":
    asyncio.run(main())
