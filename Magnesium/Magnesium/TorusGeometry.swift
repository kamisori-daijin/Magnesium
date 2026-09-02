//
//  TorusGeometry.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/09/02.
//

import Foundation
import simd

struct TorusGeometry {
    static func generateFaces() -> [[[Float16]]] {
        let majorRadius: Float = 1.5
        let minorRadius: Float = 0.6
        var faces: [[[Float16]]] = []
        
        for i in 0..<8 {
            let theta1 = Float(i) * 2.0 * Float.pi / 8.0
            let theta2 = Float(i + 1) * 2.0 * Float.pi / 8.0
            
            for j in 0..<4 {
                let phi1 = Float(j) * 2.0 * Float.pi / 4.0
                let phi2 = Float(j + 1) * 2.0 * Float.pi / 4.0
                
                let p0 = getTorusPoint(theta1, phi1, majorRadius, minorRadius)
                let p1 = getTorusPoint(theta2, phi1, majorRadius, minorRadius)
                let p2 = getTorusPoint(theta2, phi2, majorRadius, minorRadius)
                let p3 = getTorusPoint(theta1, phi2, majorRadius, minorRadius)
                
                faces.append([p0, p1, p2])
                faces.append([p0, p2, p3])
            }
        }
        return faces
    }
    
    private static func getTorusPoint(_ theta: Float, _ phi: Float, _ R: Float, _ r: Float) -> [Float16] {
        let x = (R + r * cos(phi)) * cos(theta)
        let y = r * sin(phi)
        let z = (R + r * cos(phi)) * sin(theta)
        return [Float16(x), Float16(y), Float16(z), 1.0]
    }
}
