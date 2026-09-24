import torch
import torch.nn as nn
from PreProcessor import ANE3DPreProcessor64
from ShaderModel import ANE3DRenderer64
from TextureModel import ANETextureProcessor

class ANEMonolithicPipeline(nn.Module):
    def __init__(self, target_width=1024, target_height=1024):
        super().__init__()
        self.texture_processor = ANETextureProcessor()
        self.pre_processor = ANE3DPreProcessor64()
        self.renderer = ANE3DRenderer64(target_width, target_height)
        
        # Dummy UV
        self.register_buffer("U0", torch.zeros(1, 64, 1, 1))
        self.register_buffer("V0", torch.zeros(1, 64, 1, 1))
        self.register_buffer("U1", torch.full((1, 64, 1, 1), 1.0))
        self.register_buffer("V1", torch.zeros(1, 64, 1, 1))
        self.register_buffer("U2", torch.full((1, 64, 1, 1), 0.5))
        self.register_buffer("V2", torch.full((1, 64, 1, 1), 1.0))

    def forward(self, expanded_vertices, mvp_weights, normals, light_dir, colors_r, colors_g, colors_b, raw_image):
        # 1. Texture Processing
        processed_texture = self.texture_processor(raw_image)
        
        # 2. Vertex Calculation (PreProcessor)
        # 法線(normals)を渡し、回転された法線(transformed_normals)を受け取ります
        (A0, B0, C0, A1, B1, C1, A2, B2, C2, 
         R0, G0, B0_col, R1, G1, B1_col, R2, G2, B2_col, 
         p0_iz, p1_iz, p2_iz,
         transformed_normals) = self.pre_processor(expanded_vertices, mvp_weights, normals, colors_r, colors_g, colors_b)
        
        # 3頂点分の法線に分割
        N0 = transformed_normals[:, :, 0:1, :]
        N1 = transformed_normals[:, :, 1:2, :]
        N2 = transformed_normals[:, :, 2:3, :]
        
        # 3. Rendering (Renderer)
        # 法線(N0, N1, N2)と光源(light_dir)を渡します
        R, G, B, mask_w, max_inv_z = self.renderer(
            A0, B0, C0, A1, B1, C1, A2, B2, C2,
            R0, G0, B0_col, R1, G1, B1_col, R2, G2, B2_col,
            p0_iz, p1_iz, p2_iz,
            self.U0, self.V0, self.U1, self.V1, self.U2, self.V2,
            N0, N1, N2, light_dir,
            processed_texture
        )
        
        return R, G, B, mask_w, max_inv_z