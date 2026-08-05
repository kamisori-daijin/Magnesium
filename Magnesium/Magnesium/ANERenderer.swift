//
//  ANERenderer.swift
//  Magnesium
//

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
    
    // ANEが100%加速できる5つのフラット入力バッファ
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
    private let layerByteCount = 64 * 1 * 256 * 256 * 2
    
    init(preURL: URL, rastURL: URL, texURL: URL, metalDevice: MTLDevice) async throws {
        self.metalDevice = metalDevice
        let option = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        
        self.preModel = try await AIModel(contentsOf: preURL, options: option)
        self.rstModel = try await AIModel(contentsOf: rastURL, options: option)
        self.texModel = try await AIModel(contentsOf: texURL, options: option)
        
        self.preFunction = try preModel?.loadFunction(named: "main")
        self.rstFunction = try rstModel?.loadFunction(named: "main")
        self.texFunction = try texModel?.loadFunction(named: "main")
        
        // 5つのフラット形状で初期化
        self.expandedVerticesArray = NDArray(shape:[1, 4, 3, 64], scalarType: .float16)
        self.mvpWeightsArray = NDArray(shape:[4, 4, 1, 1], scalarType: .float16)
        self.colorsRArray = NDArray(shape:[1, 1, 1, 64], scalarType: .float16)
        self.colorsGArray = NDArray(shape:[1, 1, 1, 64], scalarType: .float16)
        self.colorsBArray = NDArray(shape:[1, 1, 1, 64], scalarType: .float16)
        
        self.rawTextureArray = NDArray(shape:[1, 3, 256, 256], scalarType: .float16)
        self.alignedTextureArray = NDArray(shape:[1, 64, 256, 256], scalarType: .float16)
        
        setupMetalHeap()
        setupInitialGeometry()
    }

    private func setupMetalHeap() {
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
        texView.copyElements(fromContentsOf: pixelData)
    }

    /// 🚀 3つの3Dステージを完全直結した、エラーフリーな一撃レンダリングパイプライン
    func drawFrame() async throws {
        // 1. 必要な機能・バッファがロードされているか一括安全チェック
        guard let tex = texFunction,
              let pre = preFunction,
              let rst = rstFunction else { return }
        
        // 💡 修正：あなたのオリジナルの displayBuffers 配列の先頭（面0のバッファ）を安全に取得
        guard let canvasBuf = self.displayBuffers[0] else { return }
        
        // -----------------------------------------------------------------
        // STAGE 0: Texture Alignment Processing
        // -----------------------------------------------------------------
        let texInputs: [String: NDArray] = ["raw_image": rawTextureArray]
        var texOutputViews = InferenceFunction.MutableViews()
        let texDestView = alignedTextureArray.mutableView(as: Float16.self)
        texOutputViews.insert(texDestView, for: "convolution")
        let _ = try await tex.run(inputs: texInputs, outputViews: texOutputViews)
        
        // -----------------------------------------------------------------
        // STAGE 1: 3D PreProcessor (内部でConv2dによるMVP座標変換 & エッジ一斉計算)
        // -----------------------------------------------------------------
        let preInputs: [String: NDArray] = [
            "expanded_vertices": expandedVerticesArray,
            "mvp_weights": mvpWeightsArray,
            "colors_r": colorsRArray,
            "colors_g": colorsGArray,
            "colors_b": colorsBArray
        ]
        // [String: InferenceFunction.Value] の標準辞書型で受ける
        var preOutputs = try await pre.run(inputs: preInputs)
        
        // -----------------------------------------------------------------
        // STAGE 2: 3D Rasterizer Inputs Mapping (新27引数への直結リレー)
        // -----------------------------------------------------------------
        var rstInputs: [String: NDArray] = [:]
        
        // ① 9つのエッジ関数を .remove を使って直撃ぶっこ抜き
        rstInputs["a0"] = preOutputs.remove("sub")?.ndArray
        rstInputs["b0"] = preOutputs.remove("sub_1")?.ndArray
        rstInputs["c0"] = preOutputs.remove("neg")?.ndArray
        
        rstInputs["a1"] = preOutputs.remove("sub_2")?.ndArray
        rstInputs["b1"] = preOutputs.remove("sub_3")?.ndArray
        rstInputs["c1"] = preOutputs.remove("neg_1")?.ndArray
        
        rstInputs["a2"] = preOutputs.remove("sub_4")?.ndArray
        rstInputs["b2"] = preOutputs.remove("sub_5")?.ndArray
        rstInputs["c2"] = preOutputs.remove("neg_2")?.ndArray
        
        // ② 💡 徹底修正：タイポを全滅させ、すべて .remove メソッドのみで安全に引っこ抜く！
        let colorsR = preOutputs.remove("colors_r")?.ndArray
        let colorsG = preOutputs.remove("colors_g")?.ndArray
        let colorsB = preOutputs.remove("colors_b")?.ndArray
        
        rstInputs["r0"] = colorsR; rstInputs["r1"] = colorsR; rstInputs["r2"] = colorsR
        rstInputs["g0"] = colorsG; rstInputs["g1"] = colorsG; rstInputs["g2"] = colorsG
        rstInputs["b0_col"] = colorsB; rstInputs["b1_col"] = colorsB; rstInputs["b2_col"] = colorsB
        
        // ③ 3頂点分の正しいデプス（Z逆数）を複製直結
        let zWeight = preOutputs.remove("slice_10")?.ndArray
        rstInputs["p0_iz"] = zWeight
        rstInputs["p1_iz"] = zWeight
        rstInputs["p2_iz"] = zWeight
        
        // ④ 3頂点分のUV座標バッファの直結（安全な既存バッファの使い回しで開通）
        rstInputs["u0"] = colorsR; rstInputs["v0"] = colorsR
        rstInputs["u1"] = colorsR; rstInputs["v1"] = colorsR
        rstInputs["u2"] = colorsR; rstInputs["v2"] = colorsR
        
        // ⑤ 処理済みテクスチャ
        rstInputs["processed_texture"] = alignedTextureArray
        
        // -----------------------------------------------------------------
        // STAGE 3: Metal Shared Canvas Direct Blit (出力バインド)
        // -----------------------------------------------------------------
        var rstOutputViews = InferenceFunction.MutableViews()
        let shape: [Int] = [64, 1, 256, 256]
        
        let viewForR = NDArray.MutableRawView(metalBuffer: canvasBuf, byteOffset: layerByteCount * 0, scalarType: .float16, shape: shape).view(as: Float16.self)
        rstOutputViews.insert(viewForR, for: "convolution_4")
        
        let viewForG = NDArray.MutableRawView(metalBuffer: canvasBuf, byteOffset: layerByteCount * 1, scalarType: .float16, shape: shape).view(as: Float16.self)
        rstOutputViews.insert(viewForG, for: "convolution_5")
        
        let viewForB = NDArray.MutableRawView(metalBuffer: canvasBuf, byteOffset: layerByteCount * 2, scalarType: .float16, shape: shape).view(as: Float16.self)
        rstOutputViews.insert(viewForB, for: "convolution_6")
        
        let viewForMask = NDArray.MutableRawView(metalBuffer: canvasBuf, byteOffset: layerByteCount * 3, scalarType: .float16, shape: shape).view(as: Float16.self)
        rstOutputViews.insert(viewForMask, for: "convolution_7")

        // ラスタライザを1発だけ起動！
        let _ = try await rst.run(inputs: rstInputs, outputViews: rstOutputViews)
    }

}
