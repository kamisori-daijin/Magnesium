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
    
    internal var expandedVerticesArray: NDArray
    internal var mvpWeightsArray: NDArray
    internal var colorsRArray: NDArray
    internal var colorsGArray: NDArray
    internal var colorsBArray: NDArray
    internal var rawTextureArray: NDArray
    
    private var metalHeap: MTLHeap?
    private(set) var displayBuffers: [MTLBuffer?] = [nil, nil, nil, nil]
    
    private let metalDevice: MTLDevice
    private let layerByteCount = 1 * 1 * 1024 * 1024 * 2 // Float16 for 1024x1024
    
    // 初期化は1つのモデルURLだけでOKになります
    init(modelURL: URL, metalDevice: MTLDevice) async throws {
        self.metalDevice = metalDevice
        let option = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        
        self.pipelineModel = try await AIModel(contentsOf: modelURL, options: option)
        self.pipelineFunction = try pipelineModel?.loadFunction(named: "main")
        
        self.expandedVerticesArray = NDArray(shape:[1, 64, 4, 3], scalarType: .float16)
        self.mvpWeightsArray = NDArray(shape:[1, 64, 4, 4], scalarType: .float16)
        self.colorsRArray = NDArray(shape:[1, 64, 1, 1], scalarType: .float16)
        self.colorsGArray = NDArray(shape:[1, 64, 1, 1], scalarType: .float16)
        self.colorsBArray = NDArray(shape:[1, 64, 1, 1], scalarType: .float16)
        self.rawTextureArray = NDArray(shape:[1, 3, 256, 256], scalarType: .float16)
        
        setupMetalHeap()
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

    func drawFrame() async throws {
        guard let pipeline = pipelineFunction else { return }
        guard let canvasBuf = self.displayBuffers[0] else { return }
        
        // 1. 入力のセットアップ
        let inputs: [String: NDArray] = [
            "expanded_vertices": expandedVerticesArray,
            "mvp_weights": mvpWeightsArray,
            "colors_r": colorsRArray,
            "colors_g": colorsGArray,
            "colors_b": colorsBArray,
            "raw_image": rawTextureArray
        ]
        
        // 2. 出力先のバッファをMetalヒープに直接マッピング
        nonisolated(unsafe) var outputViews = InferenceFunction.MutableViews()
        let shape: [Int] = [1, 1, 1024, 1024]
        
        let viewForR = NDArray.MutableRawView(metalBuffer: canvasBuf, byteOffset: layerByteCount * 0, scalarType: .float16, shape: shape).view(as: Float16.self)
        outputViews.insert(viewForR, for: "upsample_bilinear2d")
        
        let viewForG = NDArray.MutableRawView(metalBuffer: canvasBuf, byteOffset: layerByteCount * 1, scalarType: .float16, shape: shape).view(as: Float16.self)
        outputViews.insert(viewForG, for: "upsample_bilinear2d_1")
        
        let viewForB = NDArray.MutableRawView(metalBuffer: canvasBuf, byteOffset: layerByteCount * 2, scalarType: .float16, shape: shape).view(as: Float16.self)
        outputViews.insert(viewForB, for: "upsample_bilinear2d_2")
        
        let viewForMask = NDArray.MutableRawView(metalBuffer: canvasBuf, byteOffset: layerByteCount * 3, scalarType: .float16, shape: shape).view(as: Float16.self)
        outputViews.insert(viewForMask, for: "upsample_bilinear2d_3")

        let viewForZ = NDArray.MutableRawView(metalBuffer: canvasBuf, byteOffset: layerByteCount * 4, scalarType: .float16, shape: shape).view(as: Float16.self)
        outputViews.insert(viewForZ, for: "upsample_bilinear2d_4")

        // 3. ANEで一括実行
        let _ = try await pipeline.run(inputs: inputs, outputViews: outputViews)
    }
}
