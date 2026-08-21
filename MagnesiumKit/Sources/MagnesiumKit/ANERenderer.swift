
import Foundation
import CoreAI
import Metal
import simd

@MainActor class ANERenderer {
    private var preModel: AIModel?
    private var rstModel: AIModel?
    private var texModel: AIModel?
    private var preFunction: InferenceFunction?
    private var rstFunction: InferenceFunction?
    private var texFunction: InferenceFunction?
    
    internal var expandedVerticesArray: NDArray
    internal var mvpWeightsArray: NDArray
    // 後半のループで不整合を防ぐため、Arrayまたは個別のNDArrayとして適切に定義
    internal var colorsRArray: NDArray
    internal var colorsGArray: NDArray
    internal var colorsBArray: NDArray
    
    internal var rawTextureArray: NDArray
    internal var alignedTextureArray: NDArray
    
    internal var metalHeap: MTLHeap?
    var displayBuffers: [MTLBuffer?] = [nil, nil, nil, nil]
    
    internal let geometry = ANE3DGeometry()
    internal let metalDevice: MTLDevice
    internal let metalCommandQueue: MTLCommandQueue
    internal let metalStream: ComputeStream
    
    // 1タイルあたり：1 * 1 * 128 * 128 * 2bytes = 32,768 bytes (32KB)
    internal let tileSizeInBytes = 1 * 1 * 128 * 128 * 2
    internal let layerByteCount = 64 * 1 * 128 * 128 * 2
    
    internal var frameIndex = 0
    internal let maxBuffersInFlight = 3
    internal var tripleOffsetsX: [[NDArray]] = []
    internal var tripleOffsetsY: [[NDArray]] = []
    
    init(preURL: URL, rastURL: URL, texURL: URL, metalDevice: MTLDevice) async throws {
        self.metalDevice = metalDevice
        // Metal CommandQueue
        self.metalCommandQueue = metalDevice.makeCommandQueue()!
        
        self.metalStream = ComputeStream(commandQueue: metalCommandQueue)
        
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
        
        setupMetalHeap()
        setupInitialGeometry()
    }
    

    // 【完全なるTBDRエミュレータ：2フェーズ分離結線システム】
    func drawFrame() async throws {
        guard let tex = texFunction, let pre = preFunction, let rst = rstFunction else { return }
        guard let canvasBuf = self.displayBuffers[frameIndex] else { return }
        
        // 🚀 Metalのタイムラインを支配するコマンドバッファを生成
        guard let commandBuffer = metalCommandQueue.makeCommandBuffer() else { return }
        
        // 事前計算用の定数定義
        let shape: [Int] = [1, 1, 128, 128]
        let singlePlaneSize = tileSizeInBytes * 192
        let localLayerByteCount = self.layerByteCount
        
        // ----------------==================================
        // 🍏 フェーズ1：タイリング（Tiling Phase）➔ 1フレームに「たった1回」だけ実行！
        // --------------------------------==================
        
        let texInputs: [String: InferenceFunction.AsyncValue] = [
            "raw_image": InferenceFunction.AsyncValue(rawTextureArray)
        ]
        
        var texOutputViews = InferenceFunction.AsyncMutableViews()
        
        // 🚀 【完全修正】すでにNDArrayがある場合は、引数にそのまま突っ込むだけで合法ラップ完了！
        var asyncTexOutput = InferenceFunction.AsyncMutableValue(alignedTextureArray)
        texOutputViews.insert(&asyncTexOutput, for: "convolution")
        
        // ストリームへ非同期エンコード（上流も下流もこれで完全に統一！）
        try tex.encode(inputs: texInputs, outputViews: texOutputViews, to: metalStream)
        
        // 2. ジオメトリ前処理
        let preInputs: [String: NDArray] = [
            "expanded_vertices": expandedVerticesArray,
            "mvp_weights": mvpWeightsArray,
            "colors_r": colorsRArray,
            "colors_g": colorsGArray,
            "colors_b": colorsBArray
        ]
        var preOutputs = try await pre.run(inputs: preInputs)
        
        // 【これが本物の Parameter Buffer だ！】
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
        // 🍏 フェーズ2：レンダリング（Rendering Phase）➔ 192個の不変の器でローテーション！
        // --------------------------------==================
        let currentXPool = tripleOffsetsX[frameIndex]
        let currentYPool = tripleOffsetsY[frameIndex]
        
        var tileCounter = 0
        for y in 0..<12 {
            for x in 0..<16 { // 2048x1536 の完全割り切り
                
                // ラスタライザ用の入力を非同期型（AsyncValue）で厳密に再構築
                var rstInputs: [String: InferenceFunction.AsyncValue] = [:]
                
                // ベースとなるジオメトリバッファを高速型ラップ
                for (key, ndArray) in baseRstInputs {
                    // 最初からオプショナルではないので、そのままAsyncValueに入れてOK！
                    rstInputs[key] = InferenceFunction.AsyncValue(ndArray)
                }

                // タイル専用の固定住所を上書き指定！
                rstInputs["tile_offset_x"] = InferenceFunction.AsyncValue(currentXPool[tileCounter])
                rstInputs["tile_offset_y"] = InferenceFunction.AsyncValue(currentYPool[tileCounter])
                
                // 💡 タイルごとに書き込みアドレス（byteOffset）を完璧にずらす計算
                let currentTileOffset = tileCounter * tileSizeInBytes
                
                // MetalBufferから「直接」非同期専用の AsyncMutableValue を生成して一括結線！
                var rstOutputViews = InferenceFunction.AsyncMutableViews()
                
                var asyncValueR = InferenceFunction.AsyncMutableValue(
                    unsafeBuffer: canvasBuf,
                    byteOffset: (singlePlaneSize * 0) + currentTileOffset,
                    scalarType: .float16,
                    shape: shape,
                    strides: [1 * 1 * 128 * 128, 128 * 128, 128, 1]
                )
                var asyncValueG = InferenceFunction.AsyncMutableValue(
                    unsafeBuffer: canvasBuf,
                    byteOffset: (singlePlaneSize * 1) + currentTileOffset,
                    scalarType: .float16,
                    shape: shape,
                    strides: [1 * 1 * 128 * 128, 128 * 128, 128, 1]
                )
                var asyncValueB = InferenceFunction.AsyncMutableValue(
                    unsafeBuffer: canvasBuf,
                    byteOffset: (singlePlaneSize * 2) + currentTileOffset,
                    scalarType: .float16,
                    shape: shape,
                    strides: [1 * 1 * 128 * 128, 128 * 128, 128, 1]
                )
                var asyncValueMask = InferenceFunction.AsyncMutableValue(
                    unsafeBuffer: canvasBuf,
                    byteOffset: (singlePlaneSize * 3) + currentTileOffset,
                    scalarType: .float16,
                    shape: shape,
                    strides: [1 * 1 * 128 * 128, 128 * 128, 128, 1]
                )
                
                // `&` を使って安全に参照渡しでインサート
                rstOutputViews.insert(&asyncValueR, for: "convolution_1")
                rstOutputViews.insert(&asyncValueG, for: "convolution_2")
                rstOutputViews.insert(&asyncValueB, for: "convolution_3")
                rstOutputViews.insert(&asyncValueMask, for: "convolution_4")
                
                // ANE非同期エンコード！
                try rst.encode(inputs: rstInputs, outputViews: rstOutputViews, to: metalStream)
                
                tileCounter += 1
            }
        }
        commandBuffer.addCompletedHandler { _ in
            // 💡 ここに到達した時点で、192枚のタイルはcanvasBufの中に完璧に整列して書き込みが終わっています！
            // このクロージャ内はスレッド安全なコンテキストなので、ここでテクスチャへの転送（Blit）などを呼び出すのが正解です。
        }
        
        // 🚀 ANEとGPUへ一斉射撃！
        commandBuffer.commit()
        
        // 次のフレームへリングバッファを回す
        frameIndex = (frameIndex + 1) % maxBuffersInFlight
    }

    
}
