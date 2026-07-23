//
//  ANE3DGeometry.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/07/19.
//

import Foundation
import simd

struct ANE3DGeometry {
    let maxVertices: Int
    
    init(maxVertices: Int = 65536) {
        self.maxVertices = maxVertices
    }
    
    /// カメラ行列（LookAt）の生成
    func createCameraMatrix(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> [Float16] {
        let zAxis = simd.normalize(eye - target)
        let xAxis = simd.normalize(simd.cross(up, zAxis))
        let yAxis = simd.cross(zAxis, xAxis)
        
        // Pythonコードと同じ計算ロジックに修正
        var R = matrix_identity_float4x4
        R.columns.0 = SIMD4<Float>(xAxis.x, xAxis.y, xAxis.z, 0)
        R.columns.1 = SIMD4<Float>(yAxis.x, yAxis.y, yAxis.z, 0)
        R.columns.2 = SIMD4<Float>(zAxis.x, zAxis.y, zAxis.z, 0)
        
        var T = matrix_identity_float4x4
        T.columns.3 = SIMD4<Float>(-eye.x, -eye.y, -eye.z, 1)
        
        // 行列の掛け算の順序を修正
        let viewMatrix = R.transpose * T
        
        var packed = [Float16](repeating: 0, count: 16)
        for i in 0..<4 {
            for j in 0..<4 {
                // 👇 [i * 4 + j] から [j * 4 + i] に変更して、NumPyの並びに合わせる
                packed[j * 4 + i] = Float16(viewMatrix[i][j])
            }
        }
        return packed
    }
    
    /// ピラミッドの頂点データを生成
    func getPyramidVertices() -> [Float16] {
        var buffer = [Float16](repeating: 0, count: 1 * 4 * 1 * maxVertices)
        
        let vertices: [[Float]] = [
            [ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, 1.0, 1.0],
            [ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, -1.0, 1.0],
            [ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, -1.0, 1.0],
            [ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0],
        ]
        
        for (i, v) in vertices.enumerated() {
            for channel in 0..<4 {
                let index = (channel * maxVertices) + i
                buffer[index] = Float16(v[channel])
            }
        }
        
        return buffer
    }
}
