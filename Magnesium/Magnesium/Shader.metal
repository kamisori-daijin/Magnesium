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

// 🌟 配列初期化構文をMSLの厳密な定義に完全修正
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
                                 constant half* buffer0 [[buffer(0)]],
                                 constant half* buffer1 [[buffer(1)]],
                                 constant half* buffer2 [[buffer(2)]],
                                 constant half* buffer3 [[buffer(3)]]) {
    uint width = 256;
    uint height = 256;
    
    uint2 coord = uint2(in.uv.x * (width - 1), (1.0 - in.uv.y) * (height - 1));
    uint pixelIndex = coord.y * width + coord.x;
    
    uint componentStride = 64 * width * height;
    
    constant half* buffers[4] = {buffer0, buffer1, buffer2, buffer3};
    half3 finalColor = half3(0.0);
    half maxMask = 0.0;
    
    for (int i = 0; i < 4; i++) {
        constant half* currentBuffer = buffers[i];
        
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
            sampledColor = clamp(sampledColor, 0.0h, 1.0h);
            
            if (mask_w > maxMask) {
                maxMask = mask_w;
                finalColor = sampledColor;
            }
        }
    }
    
    return float4(float3(finalColor), 1.0);
}
