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
    uint width = 256;
    uint height = 256;
    
    // 現在のテクスチャ座標からピクセル位置を特定
    uint2 coord = uint2(in.uv.x * (width - 1), in.uv.y * (height - 1));
    uint pixelIndex = coord.y * width + coord.x;
    
    // 💡 【重要】1レイヤー（1チャンネル）あたりの正確なピクセル数
    // ラスタライザの最終出力はポリゴン数が集約されて 1枚(1ch) になっているため、64を掛けてはいけません
    uint singleChannelStride = width * height;
    
    // Swift側の byteOffset (0, 1, 2, 3) と完全に一致するプレーンなインデックス計算
    uint rIndex = (singleChannelStride * 0) + pixelIndex;
    uint gIndex = (singleChannelStride * 1) + pixelIndex;
    uint bIndex = (singleChannelStride * 2) + pixelIndex;
    uint mIndex = (singleChannelStride * 3) + pixelIndex;
    
    half r_val = currentBuffer[rIndex];
    half g_val = currentBuffer[gIndex];
    half b_val = currentBuffer[bIndex];
    half mask_w = currentBuffer[mIndex];
    
    half4 finalColor = half4(0.0h);
    
    // Wマスクによる正規化とブレンド処理
    if (mask_w > 0.001h) {
        half3 sampledColor = half3(r_val, g_val, b_val) / (mask_w + 1e-4h);
        sampledColor = clamp(sampledColor, 0.0h, 1.0h);
        
        finalColor = half4(sampledColor, 1.0h);
    } else {
        // 背景透過
        discard_fragment();
    }
    
    return float4(finalColor);
}
