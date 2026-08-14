//
//  ANE3DGeometry+Textures.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/07/30.
//

import Foundation

extension ANE3DGeometry {
    
    public func createDebugCheckerboardTexture() -> [Float16] {
        // 128x128
        var pixelData = [Float16](repeating: 0, count: 1 * 3 * 128 * 128)
        
        for y in 0..<128 {
            for x in 0..<128 {
                let pixelIndex = y * 128 + x
                
                
                let isCenterCross = (abs(x - 64) < 2) || (abs(y - 64) < 2)
                let isGrid = (x % 16 == 0) || (y % 16 == 0)
                
                let colorValue: Float16 = (isCenterCross || isGrid) ? 1.0 : 0.0
                
                pixelData[0 * (128 * 128) + pixelIndex] = colorValue // R
                pixelData[1 * (128 * 128) + pixelIndex] = colorValue // G
                pixelData[2 * (128 * 128) + pixelIndex] = colorValue // B
            }
        }
        return pixelData
    }
}
