//
//  ANE3DGeometry+UV.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/07/30.
//

import Foundation

extension ANE3DGeometry {
    /// Make UV Data
    func getPyramidUVs() -> [Float16] {
        // [1, 2, 1, maxVertices]
        var buffer = [Float16](repeating: 0, count: 1 * 2 * 1 * maxVertices)
        
        let uvs: [[Float]] = [
            // Front
            [0.5, 1.0], [0.0, 0.0], [1.0, 0.0],
            // Right
            [0.5, 1.0], [0.0, 0.0], [1.0, 0.0],
            // Back
            [0.5, 1.0], [0.0, 0.0], [1.0, 0.0],
            // Left
            [0.5, 1.0], [0.0, 0.0], [1.0, 0.0]
        ]
        
        for (i, uv) in uvs.enumerated() {
            // maxVertices stride
            buffer[0 * maxVertices + i] = Float16(uv[0]) // U
            buffer[1 * maxVertices + i] = Float16(uv[1]) // V
        }
        
        return buffer
    }
}
