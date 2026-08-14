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
    uint tileIndex [[flat]];
};

// Vertex Shader
vertex VertexOut textureVertex(uint vertexID [[vertex_id]],
                              constant int& tileIndex [[buffer(1)]]) {
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
    
    // 15x9 grid
    int tileX = tileIndex % 15;
    int tileY = tileIndex / 15;
    
    // -1.0 〜 1.0
    float offsetX = (float(tileX) / 15.0) * 2.0 - 1.0;
    float offsetY = (float(tileY) / 9.0) * 2.0 - 1.0;
    
    VertexOut out;
    
    // Caluclate Offset
    out.position = float4(positions[vertexID].x / 15.0 + offsetX,
                          positions[vertexID].y / 9.0 + offsetY,
                          0.0, 1.0);
    out.uv = uvs[vertexID];
    out.tileIndex = tileIndex;
    
    return out;
}

// Fragment Shader
fragment float4 textureFragment(VertexOut in [[stage_in]],
                                constant half* currentBuffer [[buffer(0)]]) {
    uint width = 128;
    uint height = 128;
    
    uint2 coord = uint2(in.uv.x * (width - 1), in.uv.y * (height - 1));
    uint pixelIndex = coord.y * width + coord.x;
    
 
    uint componentStride = 64 * width * height;
    
    uint rIndex = (componentStride * 0) + pixelIndex;
    uint gIndex = (componentStride * 1) + pixelIndex;
    uint bIndex = (componentStride * 2) + pixelIndex;
    uint mIndex = (componentStride * 3) + pixelIndex;
    
    half r_val = currentBuffer[rIndex];
    half g_val = currentBuffer[gIndex];
    half b_val = currentBuffer[bIndex];
    half mask_w = currentBuffer[mIndex];
    
    if (mask_w > 0.001h) {
        half3 sampledColor = half3(r_val, g_val, b_val) / (mask_w + 1e-4h);
        return float4(float3(clamp(sampledColor, 0.0h, 1.0h)), 1.0);
    } else {
        discard_fragment();
    }
    
    return float4(0.0);
}
