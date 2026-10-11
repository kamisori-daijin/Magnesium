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
    internal var voxelDataBuffer: MTLBuffer?
    
    // Double Buffering
    private(set) var displayBuffers: [MTLBuffer] = []
    private var currentBufferIndex = 0
    
    private let metalDevice: MTLDevice
    
    private let matrixByteCount = 64 * MemoryLayout<Float16>.stride // 128 Bytes
    private let voxelByteCount = 512 * MemoryLayout<Float16>.stride // 1,024 Bytes (8x8x8)
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
        self.voxelDataBuffer = metalDevice.makeBuffer(length: voxelByteCount, options: .storageModeShared)
        
        // Double Buffering for display
        self.displayBuffers = []
        for _ in 0..<2 {
            if let buffer = metalDevice.makeBuffer(length: outputImageByteCount, options: .storageModeShared) {
                self.displayBuffers.append(buffer)
            }
        }
        
        // 初期ボクセルデータの書き込み (中心に 4x4x4 の立方体)
        if let vPtr = voxelDataBuffer?.contents().assumingMemoryBound(to: Float16.self) {
            for z in 0..<8 {
                for y in 0..<8 {
                    for x in 0..<8 {
                        let index = z * 64 + y * 8 + x
                        if (2..<6).contains(x) && (2..<6).contains(y) && (2..<6).contains(z) {
                            vPtr[index] = 1.0
                        } else {
                            vPtr[index] = 0.0
                        }
                    }
                }
            }
        }
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
        let invView = viewMatrix.inverse
        
        guard let pointer = cameraMatrixBuffer?.contents().assumingMemoryBound(to: Float16.self) else { return }

        // カメラ行列 (16要素) の書き込み
        pointer[0]  = Float16(invView.columns.0.x); pointer[1]  = Float16(invView.columns.1.x)
        pointer[2]  = Float16(invView.columns.2.x); pointer[3]  = Float16(invView.columns.3.x)
        pointer[4]  = Float16(invView.columns.0.y); pointer[5]  = Float16(invView.columns.1.y)
        pointer[6]  = Float16(invView.columns.2.y); pointer[7]  = Float16(invView.columns.3.y)
        pointer[8]  = Float16(invView.columns.0.z); pointer[9]  = Float16(invView.columns.1.z)
        pointer[10] = Float16(invView.columns.2.z); pointer[11] = Float16(invView.columns.3.z)
        pointer[12] = Float16(invView.columns.0.w); pointer[13] = Float16(invView.columns.1.w)
        pointer[14] = Float16(invView.columns.2.w); pointer[15] = Float16(invView.columns.3.w)
        
        // 残りの48要素をゼロ埋め (ANEの64並列に合わせるため)
        let zeroPointer = pointer.advanced(by: 16)
        zeroPointer.initialize(repeating: 0, count: 48)
    }

    func getCurrentDisplayBuffer() -> MTLBuffer? {
        guard !displayBuffers.isEmpty else { return nil }
        return displayBuffers[currentBufferIndex]
    }

    func drawFrame(onto stream: ComputeStream) throws {
        guard let raytracer = raytracerFunction,
              let matrixBuf = self.cameraMatrixBuffer,
              let voxelBuf = self.voxelDataBuffer,
              !displayBuffers.isEmpty else { return }
        
        let canvasBuf = displayBuffers[currentBufferIndex]

        // ANEに渡す入力データの設定
        let asyncVoxel = InferenceFunction.AsyncValue(unsafeBuffer: voxelBuf, scalarType: .float16, shape:[1, 512, 1, 1])
        let asyncMat = InferenceFunction.AsyncValue(unsafeBuffer: matrixBuf, scalarType: .float16, shape:[1, 64, 1, 1])
        
        let inputs: [String: InferenceFunction.AsyncValue] = [
            "voxel_data": asyncVoxel,
            "inv_view_matrix_64d": asyncMat
        ]
        
        var outputViews = InferenceFunction.AsyncMutableViews()
        let shape: [Int] = [1, 3, 256, 256]
        var asyncOutputValue = InferenceFunction.AsyncMutableValue(
            unsafeBuffer: canvasBuf, byteOffset: 0, scalarType: .float16, shape: shape, strides: [], interleaveLayout: nil
        )
        
        // 出力ポート名（CoreML変換時の出力名に合わせて変更してください）
        outputViews.insert(&asyncOutputValue, for: "cat_2")
        
        let _ = try raytracer.encode(inputs: inputs, outputViews: outputViews, to: stream)
        
        currentBufferIndex = (currentBufferIndex + 1) % displayBuffers.count
    }
}
