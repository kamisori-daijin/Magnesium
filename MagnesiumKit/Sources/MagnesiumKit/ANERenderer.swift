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
    private var preModel: AIModel?
    private var rstModel: AIModel?
    private var texModel: AIModel?
    
    private var preFunction: InferenceFunction?
    private var rstFunction: InferenceFunction?
    private var texFunction: InferenceFunction?
    
    internal var expandedVerticesArray: NDArray
    internal var mvpWeightsArray: NDArray
    internal var colorsRArray: NDArray
    internal var colorsGArray: NDArray
    internal var colorsBArray: NDArray
    
    internal var rawTextureArray: NDArray
    internal var alignedTextureArray: NDArray
    
    private var metalHeap: MTLHeap?
    private(set) var displayBuffers: [MTLBuffer?] = [nil, nil, nil, nil]
    
    private let geometry = ANE3DGeometry()
    private let metalDevice: MTLDevice
    
 
    private let layerByteCount = 1 * 1 * 1024 * 1024 * 2
    
    init(preURL: URL, rastURL: URL, texURL: URL, metalDevice: MTLDevice) async throws {
        self.metalDevice = metalDevice
        let option = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        
        self.preModel = try await AIModel(contentsOf: preURL, options: option)
        self.rstModel = try await AIModel(contentsOf: rastURL, options: option)
        self.texModel = try await AIModel(contentsOf: texURL, options: option)
        
        self.preFunction = try preModel?.loadFunction(named: "main")
        self.rstFunction = try rstModel?.loadFunction(named: "main")
        self.texFunction = try texModel?.loadFunction(named: "main")
        
        self.expandedVerticesArray = NDArray(shape:[1, 4, 3, 64], scalarType: .float16)
        self.mvpWeightsArray = NDArray(shape:[1, 4, 4, 1, 64], scalarType: .float16)
        self.colorsRArray = NDArray(shape:[1, 1, 1, 64], scalarType: .float16)
        self.colorsGArray = NDArray(shape:[1, 1, 1, 64], scalarType: .float16)
        self.colorsBArray = NDArray(shape:[1, 1, 1, 64], scalarType: .float16)
        
        self.rawTextureArray = NDArray(shape:[1, 3, 256, 256], scalarType: .float16)
        self.alignedTextureArray = NDArray(shape:[1, 64, 256, 256], scalarType: .float16)
        
        setupMetalHeap()
        setupInitialGeometry()
    }

    private func setupMetalHeap() {
        // R, G, B, Mask 4 Channel
        let singleDisplayBufferSize = layerByteCount * 4
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

    private func setupInitialGeometry() {
        updateGeometry()
        updateTexture()
    }

   
    func updateGeometry() {
            // Vertex Data (1 * 4 * 3 * 64 = 768)
        self.expandedVerticesArray.mutableView(as: Float16.self).withUnsafeMutablePointer { ptr, _, _ in
            geometry.writeDummyVertices(to: ptr)
        }
            
        // MVP weight (1 * 4 * 4 * 1 * 64 = 1024)
        self.mvpWeightsArray.mutableView(as: Float16.self).withUnsafeMutablePointer { ptr, _, _ in
            geometry.writeDummyMVPWeights(to: ptr)
        }
            
        // Color Data
        self.colorsRArray.mutableView(as: Float16.self).withUnsafeMutablePointer { ptr, _, _ in
            geometry.writeDummyColor(to: ptr)
        }
        self.colorsGArray.mutableView(as: Float16.self).withUnsafeMutablePointer { ptr, _, _ in
            geometry.writeDummyColor(to: ptr)
        }
        self.colorsBArray.mutableView(as: Float16.self).withUnsafeMutablePointer { ptr, _, _ in
            geometry.writeDummyColor(to: ptr)
        }
    }
        
        
    func updateTexture() {
        self.rawTextureArray.mutableView(as: Float16.self).withUnsafeMutablePointer { ptr, _, _ in
            // 1 * 3 * 256 * 256
            geometry.writeDebugCheckerboardTexture(to: ptr)
        }
    }

    func drawFrame() async throws {
        guard let tex = texFunction,
              let pre = preFunction,
              let rst = rstFunction else { return }
        
        guard let canvasBuf = self.displayBuffers[0] else { return }
        
        // STAGE 0
        let texInputs: [String: NDArray] = ["raw_image": rawTextureArray]
        var texOutputViews = InferenceFunction.MutableViews()
        let texDestView = alignedTextureArray.mutableView(as: Float16.self)
        texOutputViews.insert(texDestView, for: "convolution")
        let _ = try await tex.run(inputs: texInputs, outputViews: texOutputViews)
        
        // STAGE 1
        let preInputs: [String: NDArray] = [
            "expanded_vertices": expandedVerticesArray,
            "mvp_weights": mvpWeightsArray,
            "colors_r": colorsRArray,
            "colors_g": colorsGArray,
            "colors_b": colorsBArray
        ]
        var preOutputs = try await pre.run(inputs: preInputs)
        
        // STAGE 2
        var rstInputs: [String: NDArray] = [:]
        rstInputs["a0"] = preOutputs.remove("sub")?.ndArray
        rstInputs["b0"] = preOutputs.remove("sub_1")?.ndArray
        rstInputs["c0"] = preOutputs.remove("neg")?.ndArray
        
        rstInputs["a1"] = preOutputs.remove("sub_2")?.ndArray
        rstInputs["b1"] = preOutputs.remove("sub_3")?.ndArray
        rstInputs["c1"] = preOutputs.remove("neg_1")?.ndArray
        
        rstInputs["a2"] = preOutputs.remove("sub_4")?.ndArray
        rstInputs["b2"] = preOutputs.remove("sub_5")?.ndArray
        rstInputs["c2"] = preOutputs.remove("neg_2")?.ndArray
        
        let colorsR = preOutputs.remove("colors_r")?.ndArray
        let colorsG = preOutputs.remove("colors_g")?.ndArray
        let colorsB = preOutputs.remove("colors_b")?.ndArray
        
        rstInputs["r0"] = colorsR; rstInputs["r1"] = colorsR; rstInputs["r2"] = colorsR
        rstInputs["g0"] = colorsG; rstInputs["g1"] = colorsG; rstInputs["g2"] = colorsG
        rstInputs["b0_col"] = colorsB; rstInputs["b1_col"] = colorsB; rstInputs["b2_col"] = colorsB
        
        rstInputs["p0_iz"] = preOutputs.remove("slice_11")?.ndArray
        rstInputs["p1_iz"] = preOutputs.remove("slice_12")?.ndArray
        rstInputs["p2_iz"] = preOutputs.remove("slice_13")?.ndArray
        
        rstInputs["u0"] = colorsR; rstInputs["v0"] = colorsR
        rstInputs["u1"] = colorsR; rstInputs["v1"] = colorsR
        rstInputs["u2"] = colorsR; rstInputs["v2"] = colorsR
        
        rstInputs["processed_texture"] = alignedTextureArray
        
        // STAGE 3
        let localLayerByteCount = self.layerByteCount
        
        nonisolated(unsafe) var rstOutputViews = InferenceFunction.MutableViews()
       
        let shape: [Int] = [1, 1, 1024, 1024]
        
        let viewForR = NDArray.MutableRawView(metalBuffer: canvasBuf, byteOffset: localLayerByteCount * 0, scalarType: .float16, shape: shape).view(as: Float16.self)
        rstOutputViews.insert(viewForR, for: "upsample_bilinear2d")
        
        let viewForG = NDArray.MutableRawView(metalBuffer: canvasBuf, byteOffset: localLayerByteCount * 1, scalarType: .float16, shape: shape).view(as: Float16.self)
        rstOutputViews.insert(viewForG, for: "upsample_bilinear2d_1")
        
        let viewForB = NDArray.MutableRawView(metalBuffer: canvasBuf, byteOffset: localLayerByteCount * 2, scalarType: .float16, shape: shape).view(as: Float16.self)
        rstOutputViews.insert(viewForB, for: "upsample_bilinear2d_2")
        
        let viewForMask = NDArray.MutableRawView(metalBuffer: canvasBuf, byteOffset: localLayerByteCount * 3, scalarType: .float16, shape: shape).view(as: Float16.self)
        rstOutputViews.insert(viewForMask, for: "upsample_bilinear2d_3")

        let _ = try await rst.run(inputs: rstInputs, outputViews: rstOutputViews)
    }
}
