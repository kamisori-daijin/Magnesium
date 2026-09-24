//
//  SphereGeometry.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/09/24.
//

import Foundation
import simd

struct SphereGeometry {
    static func generateFaces() -> [[[Float16]]] {
        let radius: Float = 1.5
        var faces: [[[Float16]]] = []
        
        // 8x4 の分割で球体を作成 (合計64面)
        for i in 0..<8 {
            let theta1 = Float(i) * Float.pi / 8.0
            let theta2 = Float(i + 1) * Float.pi / 8.0
            
            for j in 0..<4 {
                let phi1 = Float(j) * 2.0 * Float.pi / 4.0
                let phi2 = Float(j + 1) * 2.0 * Float.pi / 4.0
                
                let p0 = getSpherePoint(theta1, phi1, radius)
                let p1 = getSpherePoint(theta2, phi1, radius)
                let p2 = getSpherePoint(theta2, phi2, radius)
                let p3 = getSpherePoint(theta1, phi2, radius)
                
                faces.append([p0, p1, p2])
                faces.append([p0, p2, p3])
            }
        }
        return faces
    }
    
    private static func getSpherePoint(_ theta: Float, _ phi: Float, _ r: Float) -> [Float16] {
        let x = r * sin(theta) * cos(phi)
        let y = r * cos(theta)
        let z = r * sin(theta) * sin(phi)
        return [Float16(x), Float16(y), Float16(z), 1.0]
    }
}
