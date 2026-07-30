//
//  ANE3DGeometry+UV.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/07/30.
//

import Foundation

extension ANE3DGeometry {
    /// ピラミッドの各頂点に対応するUV座標データを生成
    func getPyramidUVs() -> [Float16] {
        // [1, 2, 1, maxVertices] 形状に合わせたバッファを確保
        var buffer = [Float16](repeating: 0, count: 1 * 2 * 1 * maxVertices)
        
        let uvs: [[Float]] = [
            // 面1 (正面)
            [0.5, 1.0], [0.0, 0.0], [1.0, 0.0],
            // 面2 (右面)
            [0.5, 1.0], [0.0, 0.0], [1.0, 0.0],
            // 面3 (背面)
            [0.5, 1.0], [0.0, 0.0], [1.0, 0.0],
            // 面4 (左面)
            [0.5, 1.0], [0.0, 0.0], [1.0, 0.0]
        ]
        
        for (i, uv) in uvs.enumerated() {
            // maxVerticesのストライドでUとVを配置
            buffer[0 * maxVertices + i] = Float16(uv[0]) // U座標
            buffer[1 * maxVertices + i] = Float16(uv[1]) // V座標
        }
        
        return buffer
    }
}
