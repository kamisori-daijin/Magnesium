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
                                 constant half* buffer0 [[buffer(0)]],
                                 constant half* buffer1 [[buffer(1)]],
                                 constant half* buffer2 [[buffer(2)]],
                                 constant half* buffer3 [[buffer(3)]]) {
    uint width = 256;
    uint height = 256;
    uint2 coord = uint2(in.uv.x * (width - 1), (1.0 - in.uv.y) * (height - 1));
    uint pixelIndex = coord.y * width + coord.x;
    
    // 4つのバッファのポインタを配列にまとめる
    constant half* buffers[4] = {buffer0, buffer1, buffer2, buffer3};
    
    half3 finalColor = half3(0.0);
    
    for (int i = 0; i < 4; i++) {
        // 各バッファの0番目のチャンネルに色情報が入っていると仮定
        // （実際の色チャンネルに合わせてインデックスを調整してください）
        half val = buffers[i][pixelIndex];
        
        if (val > 0.0) {
            // 面ごとに異なる色を割り当てる（テスト用）
            half3 faceColor = half3(0.0);
            if (i == 0) faceColor.r = val; // 赤
            if (i == 1) faceColor.g = val; // 緑
            if (i == 2) faceColor.b = val; // 青
            if (i == 3) faceColor = half3(val, val, 0.0); // 黄
            
            finalColor = max(finalColor, faceColor);
        }
    }
    
    return float4(float3(finalColor), 1.0);
}
