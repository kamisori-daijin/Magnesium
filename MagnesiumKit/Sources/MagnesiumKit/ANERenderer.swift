//
//  ANERenderer.swift
//  Magnesium
//

import Foundation
import CoreAI
import Metal
import simd

@MainActor
class ANERenderer {
    private var raytracerModel: AIModel?
    private var raytracerFunction: InferenceFunction?
    
    // Input
    internal var multiviewTextureArray: NDArray
    internal var cameraMatrix64ChArray: NDArray
    
    private var metalHeap: MTLHeap?
    private(set) var displayBuffer: MTLBuffer?
    
    private let metalDevice: MTLDevice
    
    // 256 x 256 2Bytes 131,072 Bytes
    private let outputImageByteCount = 256 * 256 * 2
    
    init(raytracerURL: URL, metalDevice: MTLDevice) async throws {
        self.metalDevice = metalDevice
        let option = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        
        // 1. Load
        self.raytracerModel = try await AIModel(contentsOf: raytracerURL, options: option)
        self.raytracerFunction = try raytracerModel?.loadFunction(named: "main")
    
        // [1, 3, 256, 256]
        self.multiviewTextureArray = NDArray(shape:[1,3,256,256], scalarType: .float16)
        // [1, 64, 1, 1]
        self.cameraMatrix64ChArray = NDArray(shape:[1,64,1,1], scalarType: .float16)
        
        setupMetalHeap()
    }

    private func setupMetalHeap() {
    
        let heapDescriptor = MTLHeapDescriptor()
        heapDescriptor.size = outputImageByteCount
        heapDescriptor.storageMode = .shared
        heapDescriptor.type = .placement
        
        self.metalHeap = metalDevice.makeHeap(descriptor: heapDescriptor)
        
 
        guard let heap = self.metalHeap else { return }
        self.displayBuffer = heap.makeBuffer(
            length: outputImageByteCount,
            options: .storageModeShared,
            offset: 0
        )
    }

   
    func updateCamera(eye: simd_float3, target: simd_float3, up: simd_float3) {
        let zAxis = normalize(eye - target)
        let xAxis = normalize(cross(up, zAxis))
        let yAxis = cross(zAxis, xAxis)
        
        var R = matrix_identity_float4x4
        R.columns.0 = simd_float4(xAxis.x, yAxis.x, zAxis.x, 0.0)
        R.columns.1 = simd_float4(xAxis.y, yAxis.y, zAxis.y, 0.0)
        R.columns.2 = simd_float4(xAxis.z, yAxis.z, zAxis.z, 0.0)
        
        var T = matrix_identity_float4x4
        T.columns.3 = simd_float4(-eye.x, -eye.y, -eye.z, 1.0)
        
        let viewMatrix = matrix_multiply(R, T)
        let invView = simdfMatrixInverse(viewMatrix)
        
        cameraMatrix64ChArray.mutableView(as: Float16.self).withUnsafeMutablePointer { pointer, _, _ in

            pointer[0]  = Float16(invView.columns.0.x)
            pointer[1]  = Float16(invView.columns.1.x)
            pointer[2]  = Float16(invView.columns.2.x)
            pointer[3]  = Float16(invView.columns.3.x)
            pointer[4]  = Float16(invView.columns.0.y)
            pointer[5]  = Float16(invView.columns.1.y)
            pointer[6]  = Float16(invView.columns.2.y)
            pointer[7]  = Float16(invView.columns.3.y)
            pointer[8]  = Float16(invView.columns.0.z)
            pointer[9]  = Float16(invView.columns.1.z)
            pointer[10] = Float16(invView.columns.2.z)
            pointer[11] = Float16(invView.columns.3.z)
            pointer[12] = Float16(invView.columns.0.w)
            pointer[13] = Float16(invView.columns.1.w)
            pointer[14] = Float16(invView.columns.2.w)
            pointer[15] = Float16(invView.columns.3.w)
            
            // Padding
            let zeroPointer = pointer.advanced(by: 16)
            zeroPointer.initialize(repeating: 0, count: 48)
        }
    }



    func drawFrame() async throws {
        guard let raytracer = raytracerFunction,
              let canvasBuf = self.displayBuffer else { return }
        
        
        let inputs: [String: NDArray] = [
            "multiview_textures": multiviewTextureArray,
            "inv_view_matrix_64d": cameraMatrix64ChArray
        ]
        

        nonisolated(unsafe) var outputViews = InferenceFunction.MutableViews()
        
        let shape: [Int] = [1, 1, 256, 256]
        let destinationView = NDArray.MutableRawView(
            metalBuffer: canvasBuf,
            byteOffset: 0,
            scalarType: .float16,
            shape: shape
        ).view(as: Float16.self)
        
        outputViews.insert(destinationView, for: "mul_138")
        
 
        let _ = try await raytracer.run(inputs: inputs, outputViews: outputViews)
    }
    

    private func simdfMatrixInverse(_ m: matrix_float4x4) -> matrix_float4x4 {
        return m.inverse
    }
}
