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
    // 画面全体を覆う板ポリゴン（ここは既存のままで完璧です！）
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
    // 🌟 修正：神回路V2の出力解像度 256 x 256 にアライメント変更
    uint width = 256;
    uint height = 256;
    
    // UV座標からピクセル位置のインデックスを計算
    uint2 coord = uint2(in.uv.x * (width - 1), in.uv.y * (height - 1));
    uint pixelIndex = coord.y * width + coord.x;
    
    // 🌟 修正：ストライド地獄（R, G, B, Mask の4枚バラバラ計算）を完全全廃！
    // ANEから直撃した1チャンネルのライティンググレースケール値をそのままピンポイントで引き抜く
    half intensity = currentBuffer[pixelIndex];
    
    // 画面表示用にRGBすべてに同じ強度を割り振る（背景は真っ黒にするためそのまま代入）
    // ※if文による条件分岐や discard_fragment() はGPUのパイプラインの乱れ（遅延）を招くため、
    //  単に入力値をそのままアルファ1.0hで塗り潰すこの形が最も低負荷・爆速です！
    half3 rgbColor = clamp(half3(intensity), 0.0h, 1.0h);
    half4 finalColor = half4(rgbColor, 1.0h);
    
    return float4(finalColor);
}
