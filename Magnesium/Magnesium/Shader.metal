//
//  Shader.metal
//  Magnesium
//
//  Created by kamisori-daijin on 2026/07/14.
//

#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOut textureVertex(uint vertexID [[vertex_id]]) {
    float4 positions[4] = {
        float4(-1.0, -1.0, 0.0, 1.0),
        float4( 1.0, -1.0, 0.0, 1.0),
        float4(-1.0,  1.0, 0.0, 1.0),
        float4( 1.0,  1.0, 0.0, 1.0)
    };
    float2 uvs[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0)
    };
    
    VertexOut out;
    out.position = positions[vertexID];
    out.uv = uvs[vertexID];
    return out;
}

fragment float4 textureFragment(VertexOut in [[stage_in]],
                                 constant half* currentBuffer [[buffer(0)]]) {
    uint width = 1024;
    uint height = 1024;
    
    uint2 coord = uint2(in.uv.x * (width - 1), in.uv.y * (height - 1));
    uint pixelIndex = coord.y * width + coord.x;
    
    uint componentStride = width * height;
    
    uint rIndex = (componentStride * 0) + pixelIndex;
    uint gIndex = (componentStride * 1) + pixelIndex;
    uint bIndex = (componentStride * 2) + pixelIndex;
    uint mIndex = (componentStride * 3) + pixelIndex;
    uint zIndex = (componentStride * 4) + pixelIndex; // Zbuffer
    
    half r_val = currentBuffer[rIndex];
    half g_val = currentBuffer[gIndex];
    half b_val = currentBuffer[bIndex];
    half mask_w = currentBuffer[mIndex];
    half z_val = currentBuffer[zIndex];
    
    // Curring
    if (mask_w > 0.001h && z_val > 0.0h) {
        half3 sampledColor = half3(r_val, g_val, b_val) / (mask_w + 1e-4h);
        sampledColor = clamp(sampledColor, 0.0h, 1.0h);
        
        return float4(float3(sampledColor), 1.0h);
    } else {
        discard_fragment();
        return float4(0.0);
    }
}
