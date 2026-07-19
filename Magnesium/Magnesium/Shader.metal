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

// ANEの出力を受け取るための構造体
struct ANEPixel {
    half r, g, b, a;
};

fragment float4 textureFragment(VertexOut in [[stage_in]],
                                 constant ANEPixel* aneBuffer [[buffer(0)]]) {
    // 1024x1024の解像度を前提としたインデックス計算
    uint2 coord = uint2(in.uv.x * 1024.0, in.uv.y * 1024.0);
    uint index = coord.y * 1024 + coord.x;
    
    ANEPixel pixel = aneBuffer[index];
    
    // half (Float16) から float4 に変換して出力
    return float4(float(pixel.r), float(pixel.g), float(pixel.b), float(pixel.a));
}
