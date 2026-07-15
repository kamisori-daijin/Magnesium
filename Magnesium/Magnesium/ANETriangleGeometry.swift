//
//  ANETriangleData.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/07/11.
//

import Foundation
import simd

struct ANETriangleGeometry {
    let radius: Float
    let speed: Float
    
    init(radius: Float = 0.5, speed: Float = 2.0) {
        self.radius = radius
        self.speed = speed
    }
    
    func calculatePackedParameters(forTime time: Float) -> (weights: [Float16], biases: [Float16]) {
       
        let baseAngle = Float.pi / 2.0 - (time * speed)
        
        let angles = SIMD3<Float>(
            baseAngle,
            baseAngle - (2.0 * Float.pi / 3.0),
            baseAngle - (4.0 * Float.pi / 3.0)
        )
        
        let cosValues = simd.cos(angles)
        let sinValues = simd.sin(angles)
        
        let p0 = (x: cosValues[0] * radius, y: sinValues[0] * radius)
        let p1 = (x: cosValues[1] * radius, y: sinValues[1] * radius)
        let p2 = (x: cosValues[2] * radius, y: sinValues[2] * radius)
        
      
        @inline(__always)
        func getLineEq(pa: (x: Float, y: Float), pb: (x: Float, y: Float)) -> (A: Float, B: Float, C: Float) {
            let A = pa.y - pb.y          // Python: A = pa - pb
            let B = pb.x - pa.x          // Python: B = pb - pa
            let C = -(A * pa.x + B * pa.y) // Python: C = -(A * pa + B * pa)
            
            let length = sqrt(A * A + B * B)
            if length > 0 {
                let scale = 500.0 / length
                return (A * scale, B * scale, C * scale)
            }
            return (0, 0, 0)
        }
        
        let edge0 = getLineEq(pa: p0, pb: p1)
        let edge1 = getLineEq(pa: p1, pb: p2)
        let edge2 = getLineEq(pa: p2, pb: p0)
        
        let maxF16: Float = 65504.0
        
        @inline(__always)
        func clampAndCast(_ val: Float) -> Float16 {
            return Float16(max(-maxF16, min(maxF16, val)))
        }
        
        
        // Matches the row-major memory layout of [3, 2, 1, 1] exactly.
        // The memory layout is [edge0.A, edge0.B, edge1.A, edge1.B, edge2.A, edge2.B].
        let packedWeights: [Float16] = [
            clampAndCast(edge0.A), clampAndCast(edge0.B), // edge0
            clampAndCast(edge1.A), clampAndCast(edge1.B), // edge1
            clampAndCast(edge2.A), clampAndCast(edge2.B)  // edge2
        ]
        
       
        let biases: [Float16] = [
            clampAndCast(edge0.C),
            clampAndCast(edge1.C),
            clampAndCast(edge2.C)
        ]
        
        return (packedWeights, biases)
    }
}
