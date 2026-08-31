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

    /// [1, 64, 4, 4] 形状の頂点データを生成
        public func getDummyVertices() -> [Float16] {
            // 💡 サイズを 4 * 4 に拡張
            var buffer = [Float16](repeating: 0.0, count: 1 * 64 * 4 * 4)
            
            let pyramidFaces: [[Float]] = [
                [ 0.0,  1.0, 0.0,  -1.0, -1.0, 1.0,   1.0, -1.0, 1.0,   0.0, 0.0, 0.0], // Face0
                [ 0.0,  1.0, 0.0,   1.0, -1.0, 1.0,   1.0, -1.0, -1.0,  0.0, 0.0, 0.0], // Face1
                [ 0.0,  1.0, 0.0,   1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  0.0, 0.0, 0.0], // Face2
                [ 0.0,  1.0, 0.0,  -1.0, -1.0, -1.0, -1.0, -1.0, 1.0,   0.0, 0.0, 0.0]  // Face3
            ]
            
            for faceIdx in 0..<64 {
                let faceData = faceIdx < 4 ? pyramidFaces[faceIdx] : [Float](repeating: 0.0, count: 12)
                
                for v in 0..<4 {
                    let baseIdx = (faceIdx * 16) + (v * 4)
                    let srcIdx = v * 3
                    
                    // X, Y, Z をコピー
                    buffer[baseIdx + 0] = Float16(faceData[srcIdx + 0])
                    buffer[baseIdx + 1] = Float16(faceData[srcIdx + 1])
                    buffer[baseIdx + 2] = Float16(faceData[srcIdx + 2])
                    
                    // 💡 W座標に 1.0 を設定
                    buffer[baseIdx + 3] = (faceIdx < 4) ? 1.0 : 0.0
                }
            }
            return buffer
        }

    /// [1, 64, 4, 4] 形状のMVPウェイトを生成
    public func getDummyMVPWeights() -> [Float16] {
        var buffer = [Float16](repeating: 0.0, count: 1 * 64 * 4 * 4)
        let identity: [Float16] = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ]
        
        for faceIdx in 0..<64 {
            for i in 0..<16 {
                buffer[(faceIdx * 16) + i] = identity[i]
            }
        }
        return buffer
    }
    
    /// [1, 64, 1, 1] 形状のカラーデータを生成
    public func getDummyColor() -> [Float16] {
        var buffer = [Float16](repeating: 0.0, count: 1 * 64 * 1 * 1)
        for i in 0..<4 {
            buffer[i] = 1.0 // 最初の4つの面に色をつける
        }
        return buffer
    }
    
}
