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

// 1. 頂点シェーダーの追加
vertex VertexOut textureVertex(uint vertexID [[vertex_id]]) {
    // 画面全体を覆う三角形ストリップの座標
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

struct ANEPixel {
    half r, g, b, a;
};

// 2. フラグメントシェーダー（既存のもの）
fragment float4 textureFragment(VertexOut in [[stage_in]],
                                 constant ANEPixel* aneBuffer [[buffer(0)]]) {
    uint2 coord = uint2(in.uv.x * 1024.0, in.uv.y * 1024.0);
    uint index = coord.y * 1024 + coord.x;
    
    ANEPixel pixel = aneBuffer[index];
    
    return float4(float(pixel.r), float(pixel.g), float(pixel.b), float(pixel.a));
}
