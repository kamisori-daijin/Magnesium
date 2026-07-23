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
    
    // ループを削除し、現在のピクセルの値だけをシンプルに取得
    uint pixelIndex = coord.y * width + coord.x;
    
    half val = aneBuffer[pixelIndex];
    float normalized = float(val);
        

        
  

    
    // アルファ値にも同じ値を設定することで、Max Blendが正しく機能します
    return float4(normalized, normalized, normalized, 1.0);
}
