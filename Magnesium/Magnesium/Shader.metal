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
    
    // UV座標からピクセル位置（XY）を計算
    uint2 coord = uint2(in.uv.x * (width - 1), in.uv.y * (height - 1));
    
    // 1つのチャンネル（平面）の中でのピクセルインデックス
    uint pixelIndex = coord.y * width + coord.x;
    
    // 🌟 Planarフォーマット（R -> G -> B の順に並んでいるバッファ）からそれぞれの色をロード
    uint planeSize = width * height; // 256 * 256 = 65,536
    
    half r = currentBuffer[pixelIndex + 0 * planeSize]; // Rチャンネル面
    half g = currentBuffer[pixelIndex + 1 * planeSize]; // Gチャンネル面
    half b = currentBuffer[pixelIndex + 2 * planeSize]; // Bチャンネル面
    
    // RGBカラーをクランプして合成
    half3 rgbColor = clamp(half3(r, g, b), 0.0h, 1.0h);
    half4 finalColor = half4(rgbColor, 1.0h);
    
    return float4(finalColor);
}
