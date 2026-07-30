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
    
    // UVから解像度（256x256）のテクセル座標へマッピング
    uint2 coord = uint2(in.uv.x * (width - 1), (1.0 - in.uv.y) * (height - 1));
    uint pixelIndex = coord.y * width + coord.x;
    
    // 🌟 ANE特有のストライド計算
    // 形状 [1, 64, 256, 256] における、各チャンネル（R, G, B, Mask）プレーンの先頭要素のサイズ
    // 1プレーン = 256 * 256 = 65,536 要素
    uint planeSize = width * height;
    
    // 1つのラスタライズ成分（RやGなど）全体のサイズは 64ch 分あるので、次の成分へジャンプするストライドは：
    // 64 * 256 * 256 = 4,194,304 要素
    uint componentStride = 64 * planeSize;
    
    constant half* buffers[4] = {buffer0, buffer1, buffer2, buffer3};
    
    half3 finalColor = half3(0.0);
    half maxDepth = 0.0; // 簡易的なZバッファ（ソート）用
    
    for (int i = 0; i < 4; i++) {
        constant half* currentBuffer = buffers[i];
        
        // 🌟 ANEアライメントのデパディング抽出
        // PyTorch側で1x1 Conv合算、または拡張された際、ANEが最も得意とする「0番目のチャンネルプレーン」
        // つまり各コンポーネント（R, G, B, Mask）の先頭 [0ch * planeSize] の領域からデータを引き抜きます。
        uint rIndex = (componentStride * 0) + pixelIndex;
        uint gIndex = (componentStride * 1) + pixelIndex;
        uint bIndex = (componentStride * 2) + pixelIndex;
        uint mIndex = (componentStride * 3) + pixelIndex;
        
        half r_raw = currentBuffer[rIndex];
        half g_raw = currentBuffer[gIndex];
        half b_raw = currentBuffer[bIndex];
        half mask_w = currentBuffer[mIndex];
        
        // 🌟 ポリゴンの内側かつ有効な描画領域かを判定
        // mask_w には有効な（z_weight * mask）の値、または 1x1 Conv による合算結果が乗っています
        if (mask_w > 0.001h) {
            // 🌟 重心座標とZウェイトのデディバイド（逆変換）
            // PyTorch側で (sampled_texture * z_weight * mask) されているため、
            // mask_w (z_weight * mask) で割り算することで、純粋なテクスチャのRGBカラーを取り戻します！
            half3 sampledColor = half3(r_raw, g_raw, b_raw) / mask_w;
            
            // クランプ処理でカラーの破綻を防ぐ
            sampledColor = clamp(sampledColor, 0.0h, 1.0h);
            
            // 🌟 簡易Zバッファテスト（手前にあるポリゴンを優先描写）
            // mask_w は 1/Z（invZ）に比例しているため、値が大きいほどカメラに近い「手前」になります
            if (mask_w > maxDepth) {
                maxDepth = mask_w;
                finalColor = sampledColor;
            }
        }
    }
    
    // 画面に出力（完全なテクスチャマッピングが施された3Dピラミッドが描画されます！）
    return float4(float3(finalColor), 1.0);
}
