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
    // Porigon
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
    // 256 x 256
    uint width = 256;
    uint height = 256;
    
    
    uint2 coord = uint2(in.uv.x * (width - 1), in.uv.y * (height - 1));
    uint pixelIndex = coord.y * width + coord.x;
    
   
    half intensity = currentBuffer[pixelIndex];
    

    half3 rgbColor = clamp(half3(intensity), 0.0h, 1.0h);
    half4 finalColor = half4(rgbColor, 1.0h);
    
    return float4(finalColor);
}
