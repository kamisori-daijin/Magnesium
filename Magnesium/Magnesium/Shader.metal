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

vertex VertexOut textureVertex(uint vid [[vertex_id]]) {
    float2 positions[4] = { float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1) };
    float2 uvs[4]       = { float2(0, 1),   float2(1, 1),   float2(0, 0),   float2(1, 0) };
    
    VertexOut out;
    out.position = float4(positions[vid], 0, 1);
    out.uv = uvs[vid];
    return out;
}


fragment half4 textureFragment(VertexOut in [[stage_in]],
                               device const half* aneBuffer [[buffer(0)]])
{
    uint x = uint(in.uv.x * 1023.0f);
    uint y = uint(in.uv.y * 1023.0f);
    
    unsigned int planeSize = 1024 * 1024;
    unsigned int pixelIndex = y * 1024 + x;
    
    // Extract individual channels from CoreAI's planar output (4 channels, 4 MB total)
    half r = clamp(aneBuffer[pixelIndex + 0 * planeSize], 0.0h, 1.0h);
    half g = clamp(aneBuffer[pixelIndex + 1 * planeSize], 0.0h, 1.0h);
    half b = clamp(aneBuffer[pixelIndex + 2 * planeSize], 0.0h, 1.0h);
    half a = clamp(aneBuffer[pixelIndex + 3 * planeSize], 0.0h, 1.0h);
    
    // Directly output the RGBA data produced by the ANE as Metal pixels!
    // If the destination MTKView supports alpha transparency, the background will be automatically removed.
    return half4(r, g, b, a);
}
