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
    private var pipelineModel: AIModel?
    private var pipelineFunction: InferenceFunction?
    
    // MTLBuffer
    internal var expandedVerticesBuffer: MTLBuffer?
    internal var mvpWeightsBuffer: MTLBuffer?
    internal var colorsRBuffer: MTLBuffer?
    internal var colorsGBuffer: MTLBuffer?
    internal var colorsBBuffer: MTLBuffer?
    internal var rawTextureBuffer: MTLBuffer?
    
    private var metalHeap: MTLHeap?
    private(set) var displayBuffers: [MTLBuffer?] = [nil, nil, nil, nil]
    
    private let metalDevice: MTLDevice
    private let layerByteCount = 1 * 1 * 1024 * 1024 * 2 // Float16 for 1024x1024
    
    let sharedComputeStream: ComputeStream!
    
    init(modelURL: URL, metalDevice: MTLDevice) async throws {
        self.metalDevice = metalDevice
        let option = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        let mainMetalQueue = metalDevice.makeCommandQueue()!
        
        self.sharedComputeStream = ComputeStream(commandQueue: mainMetalQueue)
        
        self.pipelineModel = try await AIModel(contentsOf: modelURL, options: option)
        self.pipelineFunction = try pipelineModel?.loadFunction(named: "main")
        
        setupMetalBuffers()
        setupMetalHeap()
    }

    private func setupMetalBuffers() {
        let vCount = 1 * 64 * 4 * 3 * MemoryLayout<Float16>.stride
        let mCount = 1 * 64 * 4 * 4 * MemoryLayout<Float16>.stride
        let cCount = 1 * 64 * 1 * 1 * MemoryLayout<Float16>.stride
        let tCount = 1 * 3 * 256 * 256 * MemoryLayout<Float16>.stride
        
        self.expandedVerticesBuffer = metalDevice.makeBuffer(length: vCount, options: .storageModeShared)
        self.mvpWeightsBuffer = metalDevice.makeBuffer(length: mCount, options: .storageModeShared)
        self.colorsRBuffer = metalDevice.makeBuffer(length: cCount, options: .storageModeShared)
        self.colorsGBuffer = metalDevice.makeBuffer(length: cCount, options: .storageModeShared)
        self.colorsBBuffer = metalDevice.makeBuffer(length: cCount, options: .storageModeShared)
        self.rawTextureBuffer = metalDevice.makeBuffer(length: tCount, options: .storageModeShared)
    }

    private func setupMetalHeap() {
        let singleDisplayBufferSize = layerByteCount * 5
        let totalRequiredMemory = singleDisplayBufferSize * 4
        
        let heapDescriptor = MTLHeapDescriptor()
        heapDescriptor.size = totalRequiredMemory
        heapDescriptor.storageMode = .shared
        heapDescriptor.type = .placement
        
        self.metalHeap = metalDevice.makeHeap(descriptor: heapDescriptor)
        
        guard let heap = self.metalHeap else { return }
        for i in 0..<4 {
            self.displayBuffers[i] = heap.makeBuffer(
                length: singleDisplayBufferSize,
                options: .storageModeShared,
                offset: i * singleDisplayBufferSize
            )
        }
    }

    func drawFrame(onto stream: ComputeStream) throws {
        guard let pipeline = pipelineFunction else { return }
        guard let canvasBuf = self.displayBuffers[0] else { return }
        
        guard let vBuf = expandedVerticesBuffer, let mBuf = mvpWeightsBuffer,
              let rBuf = colorsRBuffer, let gBuf = colorsGBuffer, let bBuf = colorsBBuffer,
              let tBuf = rawTextureBuffer else { return }
        
        // 1.　Setup Input
        let inputs: [String: InferenceFunction.AsyncValue] = [
            "expanded_vertices": InferenceFunction.AsyncValue(unsafeBuffer: vBuf, scalarType: .float16, shape: [1, 64, 4, 3]),
            "mvp_weights": InferenceFunction.AsyncValue(unsafeBuffer: mBuf, scalarType: .float16, shape: [1, 64, 4, 4]),
            "colors_r": InferenceFunction.AsyncValue(unsafeBuffer: rBuf, scalarType: .float16, shape: [1, 64, 1, 1]),
            "colors_g": InferenceFunction.AsyncValue(unsafeBuffer: gBuf, scalarType: .float16, shape: [1, 64, 1, 1]),
            "colors_b": InferenceFunction.AsyncValue(unsafeBuffer: bBuf, scalarType: .float16, shape: [1, 64, 1, 1]),
            "raw_image": InferenceFunction.AsyncValue(unsafeBuffer: tBuf, scalarType: .float16, shape: [1, 3, 256, 256])
        ]
        
        // 2. Output Buffer to MTLHeap (AsyncMutableViews)
        var outputViews = InferenceFunction.AsyncMutableViews()
        let shape: [Int] = [1, 1, 1024, 1024]
        
        var viewForR = InferenceFunction.AsyncMutableValue(unsafeBuffer: canvasBuf, byteOffset: layerByteCount * 0, scalarType: .float16, shape: shape, strides: [], interleaveLayout: nil)
        var viewForG = InferenceFunction.AsyncMutableValue(unsafeBuffer: canvasBuf, byteOffset: layerByteCount * 1, scalarType: .float16, shape: shape, strides: [], interleaveLayout: nil)
        var viewForB = InferenceFunction.AsyncMutableValue(unsafeBuffer: canvasBuf, byteOffset: layerByteCount * 2, scalarType: .float16, shape: shape, strides: [], interleaveLayout: nil)
        var viewForMask = InferenceFunction.AsyncMutableValue(unsafeBuffer: canvasBuf, byteOffset: layerByteCount * 3, scalarType: .float16, shape: shape, strides: [], interleaveLayout: nil)
        var viewForZ = InferenceFunction.AsyncMutableValue(unsafeBuffer: canvasBuf, byteOffset: layerByteCount * 4, scalarType: .float16, shape: shape, strides: [], interleaveLayout: nil)
        
        outputViews.insert(&viewForR, for: "upsample_bilinear2d")
        outputViews.insert(&viewForG, for: "upsample_bilinear2d_1")
        outputViews.insert(&viewForB, for: "upsample_bilinear2d_2")
        outputViews.insert(&viewForMask, for: "upsample_bilinear2d_3")
        outputViews.insert(&viewForZ, for: "upsample_bilinear2d_4")

        // 3. Encode
        let _ = try pipeline.encode(inputs: inputs, outputViews: outputViews, to: stream)
    }
}
