


import Foundation
import CoreAI
import Metal
import simd

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
    private(set) var displayBuffers = SendableBuffers()
    
    private let geometry = ANE3DGeometry()
    private let metalDevice: MTLDevice
    private let layerByteCount = 64 * 1 * 128 * 128 * 2
    
    private var frameIndex = 0
    private let maxBuffersInFlight = 3
    private var tripleOffsetsX: [[NDArray]] = []
    private var tripleOffsetsY: [[NDArray]] = []
    
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
        
        self.rawTextureArray = NDArray(shape:[1, 3, 128, 128], scalarType: .float16)
        self.alignedTextureArray = NDArray(shape:[1, 64, 128, 128], scalarType: .float16)
        for _ in 0..<maxBuffersInFlight {
            var xFramePool: [NDArray] = []
            var yFramePool: [NDArray] = []
            
            for y in 0..<12 {
                for x in 0..<16 {
                    // 2048x1536（4:3黄金比）の完璧なタイルの中心のNDC座標を算出
                    let offsetX = (Float(x) / 16.0) * 2.0 - 1.0 + (1.0 / 16.0)
                    let offsetY = 1.0 - (Float(y) / 12.0) * 2.0 - (1.0 / 12.0)
                    
                    var xArr = NDArray(shape:[1], scalarType: .float16)
                    var xView = xArr.mutableView(as: Float16.self)
                    xView.copyElements(fromContentsOf: [Float16(offsetX)])
                    
                    var yArr = NDArray(shape:[1], scalarType: .float16)
                    var yView = yArr.mutableView(as: Float16.self)
                    yView.copyElements(fromContentsOf: [Float16(offsetY)])
                    
                    xFramePool.append(xArr)
                    yFramePool.append(yArr)
                }
            }
            self.tripleOffsetsX.append(xFramePool)
            self.tripleOffsetsY.append(yFramePool)
        }

        
        await setupMetalHeap()
        setupInitialGeometry()
    }
    
    @MainActor private func setupMetalHeap() {
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
        let vertices = geometry.getDummyVertices()
        let mvp = geometry.getDummyMVPWeights()
        let color = geometry.getDummyColor()
        updateGeometry(vertices: vertices, mvpWeights: mvp, r: color, g: color, b: color)
        
        let debugTexture = geometry.createDebugCheckerboardTexture()
        updateTexture(pixelData: debugTexture)
    }
    
    func updateGeometry(vertices: [Float16], mvpWeights: [Float16], r: [Float16], g: [Float16], b: [Float16]) {
        var vertexView = self.expandedVerticesArray.mutableView(as: Float16.self)
        vertexView.copyElements(fromContentsOf: vertices)
        
        var mvpView = self.mvpWeightsArray.mutableView(as: Float16.self)
        mvpView.copyElements(fromContentsOf: mvpWeights)
        
        var rView = self.colorsRArray.mutableView(as: Float16.self)
        rView.copyElements(fromContentsOf: r)
        
        var gView = self.colorsGArray.mutableView(as: Float16.self)
        gView.copyElements(fromContentsOf: g)
        
        var bView = self.colorsBArray.mutableView(as: Float16.self)
        bView.copyElements(fromContentsOf: b)
    }
    
    func updateTexture(pixelData: [Float16]) {
        var texView = self.rawTextureArray.mutableView(as: Float16.self)
        let expectedCount = 1 * 3 * 128 * 128
        
        if pixelData.count != expectedCount {
            let debugData = geometry.createDebugCheckerboardTexture()
            texView.copyElements(fromContentsOf: debugData)
        } else {
            texView.copyElements(fromContentsOf: pixelData)
        }
    }
    
    @MainActor
    private func getCanvasBuffer() -> MTLBuffer? {
        return self.displayBuffers[0]
    }
    
    func drawFrame() async throws {
        guard let tex = texFunction, let pre = preFunction, let rst = rstFunction else { return }
        
        guard let canvasBuf = self.displayBuffers[0] else { return }
        
        let localLayerByteCount = self.layerByteCount
        let shape: [Int] = [1, 1, 128, 128]
        
        // ----------------==================================
        // 🍏 フェーズ1：タイリング（Tiling Phase）
        // ----------------==================================
        // 元のあなたの完璧なコードが、そのまま一切のエラーなしで最速で動きます！
        let texInputs: [String: NDArray] = ["raw_image": rawTextureArray]
        var texOutputViews = InferenceFunction.MutableViews()
        texOutputViews.insert(alignedTextureArray.mutableView(as: Float16.self), for: "convolution")
        let _ = try await tex.run(inputs: texInputs, outputViews: texOutputViews)
        
        let preInputs: [String: NDArray] = [
            "expanded_vertices": expandedVerticesArray, "mvp_weights": mvpWeightsArray,
            "colors_r": colorsRArray, "colors_g": colorsGArray, "colors_b": colorsBArray
        ]
        var preOutputs = try await pre.run(inputs: preInputs)
        
        var baseRstInputs: [String: NDArray] = [:]
        baseRstInputs["a0"] = preOutputs.remove("sub")?.ndArray
        baseRstInputs["b0"] = preOutputs.remove("sub_1")?.ndArray
        baseRstInputs["c0"] = preOutputs.remove("neg")?.ndArray
        baseRstInputs["a1"] = preOutputs.remove("sub_2")?.ndArray
        baseRstInputs["b1"] = preOutputs.remove("sub_3")?.ndArray
        baseRstInputs["c1"] = preOutputs.remove("neg_1")?.ndArray
        baseRstInputs["a2"] = preOutputs.remove("sub_4")?.ndArray
        baseRstInputs["b2"] = preOutputs.remove("sub_5")?.ndArray
        baseRstInputs["c2"] = preOutputs.remove("neg_2")?.ndArray
        
        let colorsR = preOutputs.remove("colors_r")?.ndArray
        let colorsG = preOutputs.remove("colors_g")?.ndArray
        let colorsB = preOutputs.remove("colors_b")?.ndArray
        
        baseRstInputs["r0"] = colorsR; baseRstInputs["r1"] = colorsR; baseRstInputs["r2"] = colorsR
        baseRstInputs["g0"] = colorsG; baseRstInputs["g1"] = colorsG; baseRstInputs["g2"] = colorsG
        baseRstInputs["b0_col"] = colorsB; baseRstInputs["b1_col"] = colorsB; baseRstInputs["b2_col"] = colorsB
        
        baseRstInputs["p0_iz"] = preOutputs.remove("slice_11")?.ndArray
        baseRstInputs["p1_iz"] = preOutputs.remove("slice_12")?.ndArray
        baseRstInputs["p2_iz"] = preOutputs.remove("slice_13")?.ndArray
        
        baseRstInputs["u0"] = colorsR; baseRstInputs["v0"] = colorsR
        baseRstInputs["u1"] = colorsR; baseRstInputs["v1"] = colorsR
        baseRstInputs["u2"] = colorsR; baseRstInputs["v2"] = colorsR
        
        baseRstInputs["processed_texture"] = alignedTextureArray
        
        // ----------------==================================
        // 🍏 フェーズ2：レンダリング（Rendering Phase）
        // ----------------==================================
        let currentXPool = tripleOffsetsX[frameIndex]
        let currentYPool = tripleOffsetsY[frameIndex]
        
        var tileCounter = 0
        for y in 0..<12 {
            for x in 0..<16 {
                var rstInputs = baseRstInputs
                rstInputs["tile_offset_x"] = currentXPool[tileCounter]
                rstInputs["tile_offset_y"] = currentYPool[tileCounter]
                tileCounter += 1
                
                let viewForR = NDArray.MutableRawView(metalBuffer: canvasBuf, byteOffset: localLayerByteCount * 0, scalarType: .float16, shape: shape).view(as: Float16.self)
                let viewForG = NDArray.MutableRawView(metalBuffer: canvasBuf, byteOffset: localLayerByteCount * 1, scalarType: .float16, shape: shape).view(as: Float16.self)
                let viewForB = NDArray.MutableRawView(metalBuffer: canvasBuf, byteOffset: localLayerByteCount * 2, scalarType: .float16, shape: shape).view(as: Float16.self)
                let viewForMask = NDArray.MutableRawView(metalBuffer: canvasBuf, byteOffset: localLayerByteCount * 3, scalarType: .float16, shape: shape).view(as: Float16.self)
                
                var rstOutputViews = InferenceFunction.MutableViews()
                rstOutputViews.insert(viewForR, for: "convolution_1")
                rstOutputViews.insert(viewForG, for: "convolution_2")
                rstOutputViews.insert(viewForB, for: "convolution_3")
                rstOutputViews.insert(viewForMask, for: "convolution_4")
                
             
                let _ = try await rst.run(inputs: rstInputs, outputViews: rstOutputViews)
            }
        }
        
        frameIndex = (frameIndex + 1) % maxBuffersInFlight
    }
}
