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
        
        var R = matrix_identity_float4x4
        R.columns.0 = SIMD4<Float>(xAxis.x, yAxis.x, zAxis.x, 0)
        R.columns.1 = SIMD4<Float>(xAxis.y, yAxis.y, zAxis.y, 0)
        R.columns.2 = SIMD4<Float>(xAxis.z, yAxis.z, zAxis.z, 0)
        
        var T = matrix_identity_float4x4
        T.columns.3 = SIMD4<Float>(-simd.dot(xAxis, eye), -simd.dot(yAxis, eye), -simd.dot(zAxis, eye), 1)
        
        let viewMatrix = R * T
        
        var packed = [Float16](repeating: 0, count: 16)
        for i in 0..<4 {
            for j in 0..<4 {
                packed[i * 4 + j] = Float16(viewMatrix[j][i]) // Row-major
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
                // [1, 4, 1, maxVertices] のレイアウトに合わせたインデックス計算
                let index = (channel * maxVertices) + i
                buffer[index] = Float16(v[channel])
            }
        }
        
        return buffer
    }
}
