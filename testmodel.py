import torch
from PIL import Image
import numpy as np
import math
from MVPProcessor import ANEMVPProcessor
from ShaderModel import ANE3DRenderer64

def run_full_pipeline():
    # 1. Initialize Model
    mvp_processor = ANEMVPProcessor(max_vertices=12)
    renderer = ANE3DRenderer64(width=256, height=256)
    
    # 2. Pyramid 3D Vertex Data [1, 4, 1, 12]
    vertices_data = [
        # Front (Vertex 0,1,2)
        [ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, 1.0, 1.0],
        # Right (Vertex 3,4,5)
        [ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, -1.0, 1.0],
        # Back (Vertex 6,7,8)
        [ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, -1.0, 1.0],
        # Left (Vertex 9,10,11)
        [ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0],
    ]
    vertices = torch.tensor(vertices_data).T.unsqueeze(0).unsqueeze(2)
    
    # 3. Camera Matrix (30 rotation, 0.5 offset)
    angle = math.radians(30)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    camera_matrix = torch.tensor([
        [cos_a, 0.0, sin_a, 0.0],
        [0.0,   1.0, 0.0,  -0.5], 
        [-sin_a,0.0, cos_a, 5.0], # Z=5.0 
        [0.0,   0.0, 0.0,   1.0]
    ])
    
    # 4. Convert to 2D using MVP Processor
    with torch.no_grad():
        transformed = mvp_processor(camera_matrix, vertices)
        
    # 5. Ready to render
    final_R = torch.zeros(1, 1, 256, 256)
    final_G = torch.zeros(1, 1, 256, 256)
    final_B = torch.zeros(1, 1, 256, 256)
    
    # Color settings for each face (red, green, blue, yellow)
    colors = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0)
    ]

    # 6. Render each face
    for i in range(4):
        idx = i * 3
        p0_2d = transformed[0, :2, 0, idx].squeeze()
        p1_2d = transformed[0, :2, 0, idx+1].squeeze()
        p2_2d = transformed[0, :2, 0, idx+2].squeeze()
        
        def get_edge(p_a, p_b):
            A = p_a[1] - p_b[1]
            B = p_b[0] - p_a[0]
            C = -(A * p_a[0] + B * p_a[1])
            return A, B, C

        A0, B0, C0 = get_edge(p0_2d, p1_2d)
        A1, B1, C1 = get_edge(p1_2d, p2_2d)
        A2, B2, C2 = get_edge(p2_2d, p0_2d)

        def pack(val):
            t = torch.zeros(1, 1, 1, 64)
            t[0, 0, 0, 0] = val
            return t

        c = colors[i]
        with torch.no_grad():
            R, G, B, mask = renderer(
                pack(A0), pack(B0), pack(C0),
                pack(A1), pack(B1), pack(C1),
                pack(A2), pack(B2), pack(C2),
                pack(c[0]), pack(c[1]), pack(c[2]),
                pack(c[0]), pack(c[1]), pack(c[2]),
                pack(c[0]), pack(c[1]), pack(c[2]),
                pack(1.0 / transformed[0, 2, 0, idx])
            )
            
        
        final_R = torch.maximum(final_R, R)
        final_G = torch.maximum(final_G, G)
        final_B = torch.maximum(final_B, B)

    # 7. save image
    r_img, g_img, b_img = final_R[0, 0].numpy(), final_G[0, 0].numpy(), final_B[0, 0].numpy()
    img_array = np.stack([r_img, g_img, b_img], axis=-1)
    img_array = (np.clip(img_array, 0.0, 1.0) * 255).astype(np.uint8)
    
    Image.fromarray(img_array).save("output_pyramid.png")
    print("output_pyramid.png Saved！")

if __name__ == "__main__":
    run_full_pipeline()