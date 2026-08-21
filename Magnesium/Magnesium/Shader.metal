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
    

    int tileX = tileIndex % 16;
    int tileY = tileIndex / 16;
    
    // -1.0 〜 1.0 の位置を 16x12 の黄金比タイルに完全割り切り
    float offsetX = (float(tileX) / 16.0) * 2.0 - 1.0;
    float offsetY = (float(tileY) / 12.0) * 2.0 - 1.0;
    
    VertexOut out;
    

    out.position = float4(positions[vertexID].x / 16.0 + offsetX,
                          positions[vertexID].y / 12.0 + offsetY,
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
    

    // 1プレーン（192タイル分）の総要素数（halfの数）を正確に定義します。
    // 128 * 128 * 192 ＝ 3,145,728 要素
    uint singlePlaneElements = width * height * 192;
    
    // 現在描画しているこのタイルが、プレーンの中でどこから始まっているかのオフセット
    uint tileMemoryOffset = in.tileIndex * (width * height);
    
    // 各チャンネル（R, G, B, Mask）の、このピクセルへの絶対インデックスを算出！
    uint rIndex = (singlePlaneElements * 0) + tileMemoryOffset + pixelIndex;
    uint gIndex = (singlePlaneElements * 1) + tileMemoryOffset + pixelIndex;
    uint bIndex = (singlePlaneElements * 2) + tileMemoryOffset + pixelIndex;
    uint mIndex = (singlePlaneElements * 3) + tileMemoryOffset + pixelIndex;
    
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
    
    return float4(0.0, 0.0, 0.0, 0.0);
}

