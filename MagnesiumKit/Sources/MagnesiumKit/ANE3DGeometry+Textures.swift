//
//  ANE3DGeometry+Textures.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/07/30.
//


import Foundation

extension ANE3DGeometry {
    
    public func writeDebugCheckerboardTexture(to ptr: UnsafeMutablePointer<Float16>) {
        for y in 0..<256 {
            for x in 0..<256 {
                let pixelIndex = y * 256 + x
                
                let isCenterCross = (abs(x - 128) < 2) || (abs(y - 128) < 2)
                let isGrid = (x % 32 == 0) || (y % 32 == 0)
                
                let colorValue: Float16 = (isCenterCross || isGrid) ? 1.0 : 0.0
                
                ptr[0 * (256 * 256) + pixelIndex] = colorValue // R
                ptr[1 * (256 * 256) + pixelIndex] = colorValue // G
                ptr[2 * (256 * 256) + pixelIndex] = colorValue // B
            }
        }
    }
}
