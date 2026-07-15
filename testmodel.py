import torch
import numpy as np
from PIL import Image
from ShaderModel import ANEShaderTriangle

# 1. Initialize model (Float16)
model = ANEShaderTriangle().eval().half()

# 2. Generate 1024x1024 UV coordinates (Float16)
H, W = 1024, 1024
y_coords = torch.linspace(1.0, -1.0, H, dtype=torch.float16) 
x_coords = torch.linspace(-1.0, 1.0, W, dtype=torch.float16)
grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')


# X and Y UV coordinates
uv_input = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # Shape : (1, 2, 1024, 1024)

# 3. triangle vertex（clockwise）
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


# (A, B)、[3, 2, 1, 1] Reshape
w_flat = torch.tensor([
    A0, B0, # edge0
    A1, B1, # edge1
    A2, B2  # edge2
], dtype=torch.float16)
tri_weight = w_flat.view(3, 2, 1, 1) # Shape: (3, 2, 1, 1)

tri_bias = torch.tensor([C0, C1, C2], dtype=torch.float16)

# 5. Run Rendering
with torch.no_grad():
    output_rgba = model(uv_input, tri_weight, tri_bias) # Shape: (1, 4, 1024, 1024)

# Debug output
print("--- Tensor raw data debug ---")
print("Output shape:", output_rgba.shape)
print("R channel - Max:", output_rgba[:, 0, :, :].max().item(), "Min:", output_rgba[:, 0, :, :].min().item())

# 6. Save as image
img_data = output_rgba.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
img_data = (np.clip(img_data, 0.0, 1.0) * 255).astype(np.uint8)

img = Image.fromarray(img_data, 'RGBA')
img.save("ane_triangle_test.png")
print("Rendering complete! 'ane_triangle_test.png' is saved.")
