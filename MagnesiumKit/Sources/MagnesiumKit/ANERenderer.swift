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
    
    // Double Buffering
    private(set) var displayBuffers: [MTLBuffer] = []
    private var currentBufferIndex = 0
    
    private var metalHeap: MTLHeap?
    private let metalDevice: MTLDevice
    
    private let matrixByteCount = 64 * MemoryLayout<Float16>.stride // 128 Bytes
    private let textureByteCount = 1 * 3 * 256 * 256 * MemoryLayout<Float16>.stride // 393,216 Bytes
    private let outputImageByteCount = 3 * 256 * 256 * MemoryLayout<Float16>.stride // 393,216 Bytes
    
    let sharedComputeStream: ComputeStream!
    
    init(raytracerURL: URL, metalDevice: MTLDevice) async throws {
        self.metalDevice = metalDevice
        let option = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        let mainMetalQueue = metalDevice.makeCommandQueue()!
        
        self.sharedComputeStream = ComputeStream(commandQueue: mainMetalQueue)
        
        self.raytracerModel = try await AIModel(contentsOf: raytracerURL, options: option)
        self.raytracerFunction = try raytracerModel?.loadFunction(named: "main")
    
        setupMetalBuffers()
    }

    private func setupMetalBuffers() {
        self.cameraMatrixBuffer = metalDevice.makeBuffer(length: matrixByteCount, options: .storageModeShared)
        self.multiviewTextureBuffer = metalDevice.makeBuffer(length: textureByteCount, options: .storageModeShared)
        
        // two Buffer
        self.displayBuffers = []
        for _ in 0..<2 {
            if let buffer = metalDevice.makeBuffer(length: outputImageByteCount, options: .storageModeShared) {
                self.displayBuffers.append(buffer)
            }
        }
        
        if let tPtr = multiviewTextureBuffer?.contents().assumingMemoryBound(to: Float16.self) {
            let count = 1 * 3 * 256 * 256
            for i in 0..<count { tPtr[i] = 1.0 }
        }
    }

    func updateCamera(eye: simd_float3, target: simd_float3, up originalUp: simd_float3, time: Float) {
        let zAxis = normalize(eye - target)
        
        // 🌟【バグ修正①：天頂バグ防御レール】
        // 視線と指定したUpベクトルが平行に重なった時、カメラの横軸(xAxis)が消失するのを防ぐ
        var up = originalUp
        if abs(dot(zAxis, up)) > 0.99 {
            up = simd_float3(0.0, 0.0, 1.0)
        }
        
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
        
        // Object 1 (左側の白キューブ)
        let rotAngle1 = time * 1.5
        let scaleY1 = 1.0 + sin(time * 3.0) * 0.3
        
        // 🌟 スケール行列の初期化バグ修正 (columns.1のYだけを書き換えると他が0になり崩壊するため、正しくスケールを組み立てる)
        var modelScale1 = matrix_identity_float4x4
        modelScale1.columns.0.x = 1.0
        modelScale1.columns.1.y = scaleY1
        modelScale1.columns.2.z = 1.0
        
        var modelRot1 = matrix_identity_float4x4
        modelRot1.columns.0 = simd_float4(cos(rotAngle1), 0.0, -sin(rotAngle1), 0.0)
        modelRot1.columns.2 = simd_float4(sin(rotAngle1), 0.0, cos(rotAngle1), 0.0)
        
        var modelTrans1 = matrix_identity_float4x4
        modelTrans1.columns.3 = simd_float4(-0.6, 0.1, 0.0, 1.0)
        
        let modelMatrix1 = matrix_multiply(modelTrans1, matrix_multiply(modelRot1, modelScale1))
        let invModel1 = modelMatrix1.inverse
        
        // Object 2 (右側の青ポヨンポヨンキューブ)
        let rotAngle2 = -time * 2.0
        let posY2 = 0.1 + abs(sin(time * 4.0)) * 0.4
        var modelRot2 = matrix_identity_float4x4
        modelRot2.columns.0 = simd_float4(cos(rotAngle2), 0.0, -sin(rotAngle2), 0.0)
        modelRot2.columns.2 = simd_float4(sin(rotAngle2), 0.0, cos(rotAngle2), 0.0)
        var modelTrans2 = matrix_identity_float4x4
        modelTrans2.columns.3 = simd_float4(0.6, posY2, 0.0, 1.0)
        let modelMatrix2 = matrix_multiply(modelTrans2, modelRot2)
        let invModel2 = modelMatrix2.inverse
        
        guard let pointer = cameraMatrixBuffer?.contents().assumingMemoryBound(to: Float16.self) else { return }

        // 🌟【バグ修正②：行列データの正しい順番での書き込み（転置バグの全解消）】
        // PyTorch（get_mat_val）が期待する Row-Major（行優先）の順序にインデックスを100%正確に並び替え
        
        // [0〜15ch]: カメラの逆行列
        pointer[0]  = Float16(invView.columns.0.x); pointer[1]  = Float16(invView.columns.1.x); pointer[2]  = Float16(invView.columns.2.x); pointer[3]  = Float16(invView.columns.3.x)
        pointer[4]  = Float16(invView.columns.0.y); pointer[5]  = Float16(invView.columns.1.y); pointer[6]  = Float16(invView.columns.2.y); pointer[7]  = Float16(invView.columns.3.y)
        pointer[8]  = Float16(invView.columns.0.z); pointer[9]  = Float16(invView.columns.1.z); pointer[10] = Float16(invView.columns.2.z); pointer[11] = Float16(invView.columns.3.z)
        pointer[12] = Float16(invView.columns.0.w); pointer[13] = Float16(invView.columns.1.w); pointer[14] = Float16(invView.columns.2.w); pointer[15] = Float16(invView.columns.3.w)
        
        // [16〜31ch]: 物体1のモデル逆行列
        pointer[16] = Float16(invModel1.columns.0.x); pointer[17] = Float16(invModel1.columns.1.x); pointer[18] = Float16(invModel1.columns.2.x); pointer[19] = Float16(invModel1.columns.3.x)
        pointer[20] = Float16(invModel1.columns.0.y); pointer[21] = Float16(invModel1.columns.1.y); pointer[22] = Float16(invModel1.columns.2.y); pointer[23] = Float16(invModel1.columns.3.y)
        pointer[24] = Float16(invModel1.columns.0.z); pointer[25] = Float16(invModel1.columns.1.z); pointer[26] = Float16(invModel1.columns.2.z); pointer[27] = Float16(invModel1.columns.3.z)
        pointer[28] = Float16(invModel1.columns.0.w); pointer[29] = Float16(invModel1.columns.1.w); pointer[30] = Float16(invModel1.columns.2.w); pointer[31] = Float16(invModel1.columns.3.w)
        
        // [32〜47ch]: 物体2のモデル逆行列
        pointer[32] = Float16(invModel2.columns.0.x); pointer[33] = Float16(invModel2.columns.1.x); pointer[34] = Float16(invModel2.columns.2.x); pointer[35] = Float16(invModel2.columns.3.x)
        pointer[36] = Float16(invModel2.columns.0.y); pointer[37] = Float16(invModel2.columns.1.y); pointer[38] = Float16(invModel2.columns.2.y); pointer[39] = Float16(invModel2.columns.3.y)
        pointer[40] = Float16(invModel2.columns.0.z); pointer[41] = Float16(invModel2.columns.1.z); pointer[42] = Float16(invModel2.columns.2.z); pointer[43] = Float16(invModel2.columns.3.z)
        pointer[44] = Float16(invModel2.columns.0.w); pointer[45] = Float16(invModel2.columns.1.w); pointer[46] = Float16(invModel2.columns.2.w); pointer[47] = Float16(invModel2.columns.3.w)
        
        // ==========================================
        // [48〜53ch] マテリアル特性数値
        // ==========================================
        // アーティスト制御用の初期値に設定（PyTorchの直感パラメーター版と連動）
        pointer[48] = Float16(0.0) // 物体1透明度 (0.0=不透明)
        pointer[49] = Float16(1.0) // 物体1反射度 (1.0=完全メタル)
        pointer[50] = Float16(0.0) // 物体1歪み強度 (メタルなので0.0)
        
        // 物体2 (青)：斜め上から見てもしっかり透き通るガラス設定
        pointer[51] = Float16(1.0) // 物体2透明度：1.0 (完全透過のまま)
        pointer[52] = Float16(0.0) // 🌟反射度：0.2 ➡️ 0.0 に下げる（正面の余計なメタル感を完全に消す）
        pointer[53] = Float16(0.4) // 🌟歪み強度：0.3 ➡️ 0.4 に上げる（屈折の歪みを強調してガラスの存在感を出す）

        
        // ==========================================
        // [54〜59ch] オブジェクトのベースカラー
        // ==========================================
        // 物体1の色 (白)
        pointer[54] = Float16(1.0); pointer[55] = Float16(1.0); pointer[56] = Float16(1.0)
        
        // 物体2の色 (青)
        pointer[57] = Float16(0.3); pointer[58] = Float16(0.6); pointer[59] = Float16(1.0)
        
        // 残りの空き領域 [60〜63ch] を0でパディング
        let zeroPointer = pointer.advanced(by: 60)
        zeroPointer.initialize(repeating: 0, count: 4)
    }



    // Get Current Buffer
    func getCurrentDisplayBuffer() -> MTLBuffer? {
        guard !displayBuffers.isEmpty else { return nil }
        return displayBuffers[currentBufferIndex]
    }

    func drawFrame(onto stream: ComputeStream) throws {
        guard let raytracer = raytracerFunction,
              let matrixBuf = self.cameraMatrixBuffer,
              let texBuf = self.multiviewTextureBuffer,
              !displayBuffers.isEmpty else { return }
        
        // Set for Write destination buffer
        let canvasBuf = displayBuffers[currentBufferIndex]

        let asyncTex = InferenceFunction.AsyncValue(unsafeBuffer: texBuf, scalarType: .float16, shape:[1,3,256,256])
        let asyncMat = InferenceFunction.AsyncValue(unsafeBuffer: matrixBuf, scalarType: .float16, shape:[1,64,1,1])
        
        let inputs: [String: InferenceFunction.AsyncValue] = [
            "multiview_textures": asyncTex,
            "inv_view_matrix_64d": asyncMat
        ]
        
        var outputViews = InferenceFunction.AsyncMutableViews()
        let shape: [Int] = [1, 3, 256, 256]
        var asyncOutputValue = InferenceFunction.AsyncMutableValue(
            unsafeBuffer: canvasBuf, byteOffset: 0, scalarType: .float16, shape: shape, strides: [], interleaveLayout: nil
        )
        
        outputViews.insert(&asyncOutputValue, for: "add_160")
        
        let _ = try raytracer.encode(inputs: inputs, outputViews: outputViews, to: stream)
        
        // Set Next Bufffer
        currentBufferIndex = (currentBufferIndex + 1) % displayBuffers.count
    }
}
