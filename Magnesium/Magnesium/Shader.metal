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
    float2 uvs[4] = { float2(0.0, 1.0), float2(1.0, 1.0), float2(0.0, 0.0), float2(1.0, 0.0) };
    
    VertexOut out;
    out.position = positions[vertexID];
    out.uv = uvs[vertexID];
    return out;
}

fragment float4 textureFragment(VertexOut in [[stage_in]],
                                 constant half* aneBuffer [[buffer(0)]]) {
    uint width = 256;
    uint height = 256;
    
    uint2 coord = uint2(in.uv.x * (width - 1), (1.0 - in.uv.y) * (height - 1));
    uint pixelIndex = coord.y * width + coord.x;
    
    // 1チャンネルのデータを取得
    half gray = aneBuffer[pixelIndex];
    
    // Python側の正規化の簡易版（値が小さすぎるため、適当な係数をかけて可視化する）
    // ※ 本来はCPU側でmin/maxをとって正規化するのがベストですが、まずは表示確認用です
    float normalized = float(gray) * 100.0;
    
    // RGBすべてに同じ値を入れてグレースケールとして出力（アルファは1.0）
    return float4(normalized, normalized, normalized, 1.0);
}
