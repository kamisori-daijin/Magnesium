//
//  ANE3DGeometry.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/07/19.
//


import Foundation
import simd

public struct ANE3DGeometry {
    let maxVertices: Int
    
    public init(maxVertices: Int = 65536) {
        self.maxVertices = maxVertices
    }
    
    /// Generate LookAt camera matrix
    public func createCameraMatrix(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> [Float16] {
        let zAxis = simd.normalize(eye - target)
        let xAxis = simd.normalize(simd.cross(up, zAxis))
        let yAxis = simd.cross(zAxis, xAxis)
        
        var R = matrix_identity_float4x4
        R.columns.0 = SIMD4<Float>(xAxis.x, yAxis.x, zAxis.x, 0)
        R.columns.1 = SIMD4<Float>(xAxis.y, yAxis.y, zAxis.y, 0)
        R.columns.2 = SIMD4<Float>(xAxis.z, yAxis.z, zAxis.z, 0)
        
        var T = matrix_identity_float4x4
        T.columns.3 = SIMD4<Float>(-eye.x, -eye.y, -eye.z, 1)
        
        let viewMatrix = R * T
        
        let fov = Float.pi / 3.0
        let r = 1.0 / tan(fov * 0.5)
        let n: Float = 0.1
        let f: Float = 100.0
        
        var projMatrix = matrix_identity_float4x4
        projMatrix.columns.0 = SIMD4<Float>(r, 0, 0, 0)
        projMatrix.columns.1 = SIMD4<Float>(0, r, 0, 0)
        projMatrix.columns.2 = SIMD4<Float>(0, 0, -(f + n) / (f - n), -1)
        projMatrix.columns.3 = SIMD4<Float>(0, 0, -(2.0 * f * n) / (f - n), 0)
        
        let mvpMatrix = projMatrix * viewMatrix
        
        var packed = [Float16](repeating: 0, count: 16)
        for i in 0..<4 {
            for j in 0..<4 {
                packed[i * 4 + j] = Float16(mvpMatrix[j][i])
            }
        }
        return packed
    }

    /// Generates pyramid vertex data
    public func getPyramidVertices() -> [Float16] {
        var buffer = [Float16](repeating: 0, count: 1 * 4 * 1 * maxVertices)
        
        let vertices: [[Float]] = [
            [ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, 1.0, 1.0],
            [ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, -1.0, 1.0],
            [ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, -1.0, 1.0],
            [ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0],
        ]
        
        for (i, v) in vertices.enumerated() {
            buffer[0 * maxVertices + i] = Float16(v[0])
            buffer[1 * maxVertices + i] = Float16(v[1])
            buffer[2 * maxVertices + i] = Float16(v[2])
            buffer[3 * maxVertices + i] = Float16(v[3])
        }
        
        return buffer
    }

    // [1, 4, 3, 64]
    public func writeDummyVertices(to ptr: UnsafeMutablePointer<Float16>) {
        let maxFaces = 64
        let totalCount = 1 * 4 * 3 * 64
        
        for i in 0..<totalCount { ptr[i] = 0.0 }
        
        let wChannelOffset = 3 * 3 * maxFaces
        
        for faceIdx in 0..<maxFaces {
            ptr[wChannelOffset + (0 * maxFaces) + faceIdx] = 1.0
            ptr[wChannelOffset + (1 * maxFaces) + faceIdx] = 1.0
            ptr[wChannelOffset + (2 * maxFaces) + faceIdx] = 1.0
        }
    }

    public func writeDummyMVPWeights(to ptr: UnsafeMutablePointer<Float16>) {
        for i in 0..<(4 * 4) { ptr[i] = 0.0 }
        ptr[0]  = 1.0
        ptr[5]  = 1.0
        ptr[10] = 1.0
        ptr[15] = 1.0
    }

    public func writeDummyColor(to ptr: UnsafeMutablePointer<Float16>) {
        for i in 0..<64 { ptr[i] = 0.0 }
    }
}
