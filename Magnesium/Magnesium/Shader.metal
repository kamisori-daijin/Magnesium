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
    
    half maxVal = 0.0;
    
    // ストライド [4194304, 65536, 256, 1] に合わせたインデックス計算
    // 65536 は 256*256 なので、チャンネルごとのオフセットです
    for (uint c = 0; c < 64; ++c) {
        uint channelOffset = c * 65536;
        uint yOffset = coord.y * 256;
        uint xOffset = coord.x;
        
        uint pixelIndex = channelOffset + yOffset + xOffset;
        
        half val = aneBuffer[pixelIndex];
        if (val > maxVal) {
            maxVal = val;
        }
    }
    
    float normalized = float(maxVal);
    return float4(normalized, normalized, normalized, 1.0);
}
