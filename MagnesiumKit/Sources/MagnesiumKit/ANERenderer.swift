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
    private let outputImageByteCount = 3 * 256 * 256 * MemoryLayout<Float16>.stride // 393,216 Bytes
    
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
        
        // ==========================================
        // 📦 オブジェクト1の行列計算 (左側に配置、Y軸を伸縮させて回転)
        // ==========================================
        let rotAngle1 = time * 1.5
        let scaleY1 = 1.0 + sin(time * 3.0) * 0.3
        
        var modelRot1 = matrix_identity_float4x4
        modelRot1.columns.0 = simd_float4(cos(rotAngle1), 0.0, -sin(rotAngle1), 0.0)
        modelRot1.columns.2 = simd_float4(sin(rotAngle1), 0.0, cos(rotAngle1), 0.0)
        
        var modelScale1 = matrix_identity_float4x4
        modelScale1.columns.1 = simd_float4(0.0, scaleY1, 0.0, 0.0)
        
        var modelTrans1 = matrix_identity_float4x4
        modelTrans1.columns.3 = simd_float4(-0.6, 0.1, 0.0, 1.0) // 🌟左側に -0.6 ずらす
        
        let modelMatrix1 = matrix_multiply(modelTrans1, matrix_multiply(modelRot1, modelScale1))
        let invModel1 = modelMatrix1.inverse
        
        // ==========================================
        // 📦 オブジェクト2の行列計算 (右側に配置、逆回転、上下に弾む)
        // ==========================================
        let rotAngle2 = -time * 2.0 // 🌟逆回転
        let posY2 = 0.1 + abs(sin(time * 4.0)) * 0.4 // 🌟上下にバウンド
        
        var modelRot2 = matrix_identity_float4x4
        modelRot2.columns.0 = simd_float4(cos(rotAngle2), 0.0, -sin(rotAngle2), 0.0)
        modelRot2.columns.2 = simd_float4(sin(rotAngle2), 0.0, cos(rotAngle2), 0.0)
        
        var modelTrans2 = matrix_identity_float4x4
        modelTrans2.columns.3 = simd_float4(0.6, posY2, 0.0, 1.0) // 🌟右側に 0.6 ずらす
        
        let modelMatrix2 = matrix_multiply(modelTrans2, modelRot2)
        let invModel2 = modelMatrix2.inverse
        
        guard let pointer = cameraMatrixBuffer?.contents().assumingMemoryBound(to: Float16.self) else { return }

        // [0〜15ch]: カメラの逆行列
        pointer[0]  = Float16(invView.columns.0.x); pointer[1]  = Float16(invView.columns.1.x)
        pointer[2]  = Float16(invView.columns.2.x); pointer[3]  = Float16(invView.columns.3.x)
        pointer[4]  = Float16(invView.columns.0.y); pointer[5]  = Float16(invView.columns.1.y)
        pointer[6]  = Float16(invView.columns.2.y); pointer[7]  = Float16(invView.columns.3.y)
        pointer[8]  = Float16(invView.columns.0.z); pointer[9]  = Float16(invView.columns.1.z)
        pointer[10] = Float16(invView.columns.2.z); pointer[11] = Float16(invView.columns.3.z)
        pointer[12] = Float16(invView.columns.0.w); pointer[13] = Float16(invView.columns.1.w)
        pointer[14] = Float16(invView.columns.2.w); pointer[15] = Float16(invView.columns.3.w)
        
        // [16〜31ch]: オブジェクト1の逆行列
        pointer[16] = Float16(invModel1.columns.0.x); pointer[17] = Float16(invModel1.columns.1.x)
        pointer[18] = Float16(invModel1.columns.2.x); pointer[19] = Float16(invModel1.columns.3.x)
        pointer[20] = Float16(invModel1.columns.0.y); pointer[21] = Float16(invModel1.columns.1.y)
        pointer[22] = Float16(invModel1.columns.2.y); pointer[23] = Float16(invModel1.columns.3.y)
        pointer[24] = Float16(invModel1.columns.0.z); pointer[25] = Float16(invModel1.columns.1.z)
        pointer[26] = Float16(invModel1.columns.2.z); pointer[27] = Float16(invModel1.columns.3.z)
        pointer[28] = Float16(invModel1.columns.0.w); pointer[29] = Float16(invModel1.columns.1.w)
        pointer[30] = Float16(invModel1.columns.2.w); pointer[31] = Float16(invModel1.columns.3.w)
        
        // 🌟 [32〜47ch]: オブジェクト2の逆行列を書き込み！
        pointer[32] = Float16(invModel2.columns.0.x); pointer[33] = Float16(invModel2.columns.1.x)
        pointer[34] = Float16(invModel2.columns.2.x); pointer[35] = Float16(invModel2.columns.3.x)
        pointer[36] = Float16(invModel2.columns.0.y); pointer[37] = Float16(invModel2.columns.1.y)
        pointer[38] = Float16(invModel2.columns.2.y); pointer[39] = Float16(invModel2.columns.3.y)
        pointer[40] = Float16(invModel2.columns.0.z); pointer[41] = Float16(invModel2.columns.1.z)
        pointer[42] = Float16(invModel2.columns.2.z); pointer[43] = Float16(invModel2.columns.3.z)
        pointer[44] = Float16(invModel2.columns.0.w); pointer[45] = Float16(invModel2.columns.1.w)
        pointer[46] = Float16(invModel2.columns.2.w); pointer[47] = Float16(invModel2.columns.3.w)
        
        // 残りの空き領域 [48〜63ch] のみを0で埋める
        let zeroPointer = pointer.advanced(by: 48)
        zeroPointer.initialize(repeating: 0, count: 16)
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
        let shape: [Int] = [1, 3, 256, 256]
        var asyncOutputValue = InferenceFunction.AsyncMutableValue(
            unsafeBuffer: canvasBuf,
            byteOffset: 0,
            scalarType: .float16,
            shape: shape,
            strides: [],
            interleaveLayout: nil
        )
        
        outputViews.insert(&asyncOutputValue, for: "mul_342")
        
        
        let _ = try raytracer.encode(inputs: inputs, outputViews: outputViews, to: stream)
    }
}
