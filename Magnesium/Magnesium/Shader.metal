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
    uint width = 256;
    uint height = 256;
    
    // ピクセル座標の計算
    uint2 coord = uint2(in.uv.x * (width - 1), in.uv.y * (height - 1));
    
    // 1プレーン（1色分）のサイズ
    uint planeSize = width * height;
    
    // 正しいPlanar形式のインデックス計算
    // coord.y * width + coord.x でその色の中のピクセル位置を出す
    uint pixelIndex = coord.y * width + coord.x;
    
    half r = currentBuffer[pixelIndex + 0 * planeSize];
    half g = currentBuffer[pixelIndex + 1 * planeSize];
    half b = currentBuffer[pixelIndex + 2 * planeSize];
    
    half3 rgbColor = clamp(half3(r, g, b), 0.0h, 1.0h);
    
    return float4(float3(rgbColor), 1.0);
}
