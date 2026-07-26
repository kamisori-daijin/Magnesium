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
    
    /// Generate LookAt camera matrix
    func createCameraMatrix(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> [Float16] {
        let zAxis = simd.normalize(eye - target)
        let xAxis = simd.normalize(simd.cross(up, zAxis))
        let yAxis = simd.cross(zAxis, xAxis)
        
        // Build row-major matrix data
        var R = matrix_identity_float4x4
        R.columns.0 = SIMD4<Float>(xAxis.x, yAxis.x, zAxis.x, 0)
        R.columns.1 = SIMD4<Float>(xAxis.y, yAxis.y, zAxis.y, 0)
        R.columns.2 = SIMD4<Float>(xAxis.z, yAxis.z, zAxis.z, 0)
        
        var T = matrix_identity_float4x4
        T.columns.3 = SIMD4<Float>(-eye.x, -eye.y, -eye.z, 1)
        
        // Align with Python's R @ T multiplication
        let viewMatrix = R * T
        
        var packed = [Float16](repeating: 0, count: 16)
        for i in 0..<4 {
            for j in 0..<4 {
                packed[i * 4 + j] = Float16(viewMatrix[j][i])
            }
        }
        return packed
    }
    
    /// Generates pyramid vertex data
    func getPyramidVertices() -> [Float16] {
        var buffer = [Float16](repeating: 0, count: 1 * 4 * 1 * maxVertices)
        
        let vertices: [[Float]] = [
            [ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, 1.0, 1.0],
            [ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, -1.0, 1.0],
            [ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, -1.0, 1.0],
            [ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0],
        ]
        
        for (i, v) in vertices.enumerated() {
            // Space elements by maxVertices
            buffer[0 * maxVertices + i] = Float16(v[0]) // X
            buffer[1 * maxVertices + i] = Float16(v[1]) // Y
            buffer[2 * maxVertices + i] = Float16(v[2]) // Z
            buffer[3 * maxVertices + i] = Float16(v[3]) // W
        }
        
        return buffer
    }
}
