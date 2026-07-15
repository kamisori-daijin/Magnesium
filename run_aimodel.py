import asyncio
from pathlib import Path
import numpy as np
from PIL import Image

from coreai.authoring import AIModelAsset
from coreai.runtime import InferenceFunction, NDArray

async def main():
    asset_path = Path("./shader_triangle.aimodel")
    
    if not asset_path.exists():
        print(f"Error: {asset_path} not found.")
        return

    print("Loading AIModel Asset...")
    asset = AIModelAsset.load(asset_path)
    
    async with asset.executable() as model:
        function: InferenceFunction = model.load_function("main")
        desc = function.desc

        # -----------------------------------------------------------
        # 1. make input data (no padding, 2 channels only)
        # -----------------------------------------------------------
        H, W = 1024, 1024
        
        y_coords = np.linspace(1.0, -1.0, H, dtype=np.float16)
        x_coords = np.linspace(-1.0, 1.0, W, dtype=np.float16)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        
        # X and Y only (2 channels)
        uv_data = np.stack([grid_x, grid_y], axis=0)[np.newaxis, ...] # Shape: (1, 2, 1024, 1024)

        # 2. calculate triangle vertices
        p0, p1, p2 = (0.0, 0.6), (0.5, -0.4), (-0.5, -0.4)
        
        def get_line_eq(pa, pb):
            A = pa[1] - pb[1]
            B = pb[0] - pa[0]
            C = -(A * pa[0] + B * pa[1])
            
            length = (A**2 + B**2)**0.5
            if length > 0:
                A, B, C = A / length, B / length, C / length
                
            edge_sharpness = 500.0
            return A * edge_sharpness, B * edge_sharpness, C * edge_sharpness

        A0, B0, C0 = get_line_eq(p0, p1)
        A1, B1, C1 = get_line_eq(p1, p2)
        A2, B2, C2 = get_line_eq(p2, p0)

        # No padding (3, 2, 1, 1) weights
        weight_flat = [A0, B0, A1, B1, A2, B2]
        weight_data = np.array(weight_flat, dtype=np.float16).reshape(3, 2, 1, 1)
        bias_data = np.array([C0, C1, C2], dtype=np.float16)

        inputs = {
            "x": NDArray(uv_data),
            "tri_weight": NDArray(weight_data),
            "tri_bias": NDArray(bias_data)
        }

        print("🚀 Running Inference...")
        outputs = await function(inputs)
        
        output_key = desc.output_names[0]
        result = outputs[output_key].numpy()

    # -----------------------------------------------------------
    # 2. Save as RGBA image
    # -----------------------------------------------------------
    print("📸 Inference completed. Saving as RGBA image...")
    
    # Change Shape (4, 1024, 1024) to (1024, 1024, 4)
    img_data = np.squeeze(result)
    if img_data.shape[0] == 4:
        img_data = np.transpose(img_data, (1, 2, 0))

    # 0.0〜1.0 value 0〜255 (uint8) clamp convert
    final_img_data = (np.clip(img_data, 0.0, 1.0) * 255).astype(np.uint8)

    # Write as RGBA image
    img = Image.fromarray(final_img_data, 'RGBA')
    img.save("coreai_pure_test.png")
    
    print("'coreai_pure_test.png' saved！")

if __name__ == "__main__":
    asyncio.run(main())
