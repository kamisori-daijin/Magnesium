//
//  ANE3DGeometry+Textures.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/07/30.
//

import Foundation

extension ANE3DGeometry {
    /// デバッグ用の256x256のチェッカーボード（白黒市松模様）テクスチャ配列を生成
    func createDebugCheckerboardTexture() -> [Float16] {
        var pixelData = [Float16](repeating: 0, count: 1 * 3 * 256 * 256) // [1, 3, 256, 256]
        
        for y in 0..<256 {
            for x in 0..<256 {
                // 32ピクセルごとの格子模様を計算
                let isWhite = ((x / 32) + (y / 32)) % 2 == 0
                let colorValue: Float16 = isWhite ? 1.0 : 0.0
                
                let pixelIndex = y * 256 + x
                
                // PyTorchの[Channel, H, W]のプレーン配置に合わせる
                pixelData[0 * (256 * 256) + pixelIndex] = colorValue // R
                pixelData[1 * (256 * 256) + pixelIndex] = colorValue // G
                pixelData[2 * (256 * 256) + pixelIndex] = colorValue // B
            }
        }
        return pixelData
    }
}
