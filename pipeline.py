import torch
import torch.nn as nn
from PreProcessor import ANE3DPreProcessor
from ShaderModel import ANE3DRenderer
from TextureModel import ANETextureProcessor

class ANEMonolithicPipeline(nn.Module):
    def __init__(self, target_width=1024, target_height=1024, max_polygons=256):
        super().__init__()
        self.max_polygons = max_polygons
        
        self.texture_processor = ANETextureProcessor(max_polygons=max_polygons)
        self.pre_processor = ANE3DPreProcessor(max_polygons=max_polygons)
        self.renderer = ANE3DRenderer(target_width, target_height, max_polygons=max_polygons)
        
        # Dummy UV (max_polygons に対応)
        self.register_buffer("U0", torch.zeros(1, self.max_polygons, 1, 1))
        self.register_buffer("V0", torch.zeros(1, self.max_polygons, 1, 1))
        self.register_buffer("U1", torch.full((1, self.max_polygons, 1, 1), 1.0))
        self.register_buffer("V1", torch.zeros(1, self.max_polygons, 1, 1))
        self.register_buffer("U2", torch.full((1, self.max_polygons, 1, 1), 0.5))
        self.register_buffer("V2", torch.full((1, self.max_polygons, 1, 1), 1.0))

    def forward(self, expanded_vertices, mvp_weights, colors_r, colors_g, colors_b, raw_image):
        processed_texture = self.texture_processor(raw_image)
        
        (A0, B0, C0, A1, B1, C1, A2, B2, C2, 
         R0, G0, B0_col, R1, G1, B1_col, R2, G2, B2_col, 
         p0_iz, p1_iz, p2_iz) = self.pre_processor(expanded_vertices, mvp_weights, colors_r, colors_g, colors_b)
        
        R, G, B, mask_w, max_inv_z = self.renderer(
            A0, B0, C0, A1, B1, C1, A2, B2, C2,
            R0, G0, B0_col, R1, G1, B1_col, R2, G2, B2_col,
            p0_iz, p1_iz, p2_iz,
            self.U0, self.V0, self.U1, self.V1, self.U2, self.V2,
            processed_texture
        )
        
        return R, G, B, mask_w, max_inv_z