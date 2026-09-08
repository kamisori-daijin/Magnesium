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
    

    internal var cameraMatrixBuffer: MTLBuffer?
    internal var multiviewTextureBuffer: MTLBuffer?
    private(set) var displayBuffer: MTLBuffer?
    
    private var metalHeap: MTLHeap?
    private let metalDevice: MTLDevice
    
    private let matrixByteCount = 64 * MemoryLayout<Float16>.stride // 128 Bytes
    private let textureByteCount = 1 * 3 * 256 * 256 * MemoryLayout<Float16>.stride // 393,216 Bytes
    private let outputImageByteCount = 256 * 256 * MemoryLayout<Float16>.stride // 131,072 Bytes
    
    // Stream
    let sharedComputeStream: ComputeStream!
    
    init(raytracerURL: URL, metalDevice: MTLDevice) async throws {
        self.metalDevice = metalDevice
        let option = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        let mainMetalQueue = metalDevice.makeCommandQueue()!
        
        self.sharedComputeStream = ComputeStream(commandQueue: mainMetalQueue)
        
        // 1. Load
        self.raytracerModel = try await AIModel(contentsOf: raytracerURL, options: option)
        self.raytracerFunction = try raytracerModel?.loadFunction(named: "main")
    
        setupMetalBuffers()
    }

    private func setupMetalBuffers() {
 
        self.cameraMatrixBuffer = metalDevice.makeBuffer(length: matrixByteCount, options: .storageModeShared)
        self.multiviewTextureBuffer = metalDevice.makeBuffer(length: textureByteCount, options: .storageModeShared)
        self.displayBuffer = metalDevice.makeBuffer(length: outputImageByteCount, options: .storageModeShared)
        
        // White
        if let tPtr = multiviewTextureBuffer?.contents().assumingMemoryBound(to: Float16.self) {
            let count = 1 * 3 * 256 * 256
            for i in 0..<count { tPtr[i] = 1.0 }
        }
    }

    func updateCamera(eye: simd_float3, target: simd_float3, up: simd_float3, time: Float) {
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
        let invView = viewMatrix.inverse
        
        // Inverse calculation
        let rotationAngle = time * 1.5
        let scaleY = 1.0 + sin(time * 3.0) * 0.3 // Y Axis Compression
        
        var modelRot = matrix_identity_float4x4
        modelRot.columns.0 = simd_float4(cos(rotationAngle), 0.0, -sin(rotationAngle), 0.0)
        modelRot.columns.2 = simd_float4(sin(rotationAngle), 0.0, cos(rotationAngle), 0.0)
        
        var modelScale = matrix_identity_float4x4
        modelScale.columns.1 = simd_float4(0.0, scaleY, 0.0, 0.0)
        
        var modelTrans = matrix_identity_float4x4
        modelTrans.columns.3 = simd_float4(0.0, 0.1, 0.0, 1.0)
        
        let modelMatrix = matrix_multiply(modelTrans, matrix_multiply(modelRot, modelScale))
        let invModel = modelMatrix.inverse
        
     
        guard let pointer = cameraMatrixBuffer?.contents().assumingMemoryBound(to: Float16.self) else { return }


        // [0〜15ch]: Camera Inverse matrix
        pointer[0]  = Float16(invView.columns.0.x); pointer[1]  = Float16(invView.columns.1.x)
        pointer[2]  = Float16(invView.columns.2.x); pointer[3]  = Float16(invView.columns.3.x)
        pointer[4]  = Float16(invView.columns.0.y); pointer[5]  = Float16(invView.columns.1.y)
        pointer[6]  = Float16(invView.columns.2.y); pointer[7]  = Float16(invView.columns.3.y)
        pointer[8]  = Float16(invView.columns.0.z); pointer[9]  = Float16(invView.columns.1.z)
        pointer[10] = Float16(invView.columns.2.z); pointer[11] = Float16(invView.columns.3.z)
        pointer[12] = Float16(invView.columns.0.w); pointer[13] = Float16(invView.columns.1.w)
        pointer[14] = Float16(invView.columns.2.w); pointer[15] = Float16(invView.columns.3.w)
        
        // [16〜31ch]
        pointer[16] = Float16(invModel.columns.0.x); pointer[17] = Float16(invModel.columns.1.x)
        pointer[18] = Float16(invModel.columns.2.x); pointer[19] = Float16(invModel.columns.3.x)
        pointer[20] = Float16(invModel.columns.0.y); pointer[21] = Float16(invModel.columns.1.y)
        pointer[22] = Float16(invModel.columns.2.y); pointer[23] = Float16(invModel.columns.3.y)
        pointer[24] = Float16(invModel.columns.0.z); pointer[25] = Float16(invModel.columns.1.z)
        pointer[26] = Float16(invModel.columns.2.z); pointer[27] = Float16(invModel.columns.3.z)
        pointer[28] = Float16(invModel.columns.0.w); pointer[29] = Float16(invModel.columns.1.w)
        pointer[30] = Float16(invModel.columns.2.w); pointer[31] = Float16(invModel.columns.3.w)
        
        // 0 Padding
        let zeroPointer = pointer.advanced(by: 32)
        zeroPointer.initialize(repeating: 0, count: 32)
    }


 
    func drawFrame(onto stream: ComputeStream) throws {
        guard let raytracer = raytracerFunction,
              let matrixBuf = self.cameraMatrixBuffer,
              let texBuf = self.multiviewTextureBuffer,
              let canvasBuf = self.displayBuffer else { return }
        
 

        let asyncTex = InferenceFunction.AsyncValue(
            unsafeBuffer: texBuf,
            scalarType: .float16,
            shape:[1,3,256,256]
        )
        let asyncMat = InferenceFunction.AsyncValue(
            unsafeBuffer: matrixBuf,
            scalarType: .float16,
            shape:[1,64,1,1]
        )
        
        let inputs: [String: InferenceFunction.AsyncValue] = [
            "multiview_textures": asyncTex,
            "inv_view_matrix_64d": asyncMat
        ]
        
        var outputViews = InferenceFunction.AsyncMutableViews()
        let shape: [Int] = [1, 1, 256, 256]
        var asyncOutputValue = InferenceFunction.AsyncMutableValue(
            unsafeBuffer: canvasBuf,
            byteOffset: 0,
            scalarType: .float16,
            shape: shape,
            strides: [],
            interleaveLayout: nil
        )
        
        outputViews.insert(&asyncOutputValue, for: "mul_192")
        
        
        let _ = try raytracer.encode(inputs: inputs, outputViews: outputViews, to: stream)
    }
}
