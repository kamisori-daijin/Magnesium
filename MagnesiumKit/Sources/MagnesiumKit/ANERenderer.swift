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
    
    // ANEへの入力用NDArray（Swift側でテクスチャデータを直接更新できるポート）
    internal var multiviewTextureArray: NDArray
    internal var cameraMatrix64ChArray: NDArray
    
    private var metalHeap: MTLHeap?
    // レンダリング結果がダイレクトに書き込まれるMetalバッファ
    private(set) var displayBuffer: MTLBuffer?
    
    private let metalDevice: MTLDevice
    
    // 256 x 256 の画像、Float16(2バイト) の総バイト数 (131,072 バイト)
    private let outputImageByteCount = 256 * 256 * 2
    
    init(raytracerURL: URL, metalDevice: MTLDevice) async throws {
        self.metalDevice = metalDevice
        
        // 完璧にANEを縛り付けるオプション設定
        let option = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        
        // 1. 新しい神回路モデル（.aimodel）のロード
        self.raytracerModel = try await AIModel(contentsOf: raytracerURL, options: option)
        self.raytracerFunction = try raytracerModel?.loadFunction(named: "main")
        
        // 2. 入力テンソルの形状を【完全ANEアライメント】で初期化
        // 3面図テクスチャ [1, 3, 256, 256]
        self.multiviewTextureArray = NDArray(shape:[1,3,256,256], scalarType: .float16)
        // 64チャンネル化されたカメラ逆行列 [1, 64, 1, 1]
        self.cameraMatrix64ChArray = NDArray(shape:[1,64,1,1], scalarType: .float16)
        
        setupMetalHeap()
    }

    private func setupMetalHeap() {
        // 出力画像に必要なメモリサイズでMetalヒープをカチッと確保
        let heapDescriptor = MTLHeapDescriptor()
        heapDescriptor.size = outputImageByteCount
        heapDescriptor.storageMode = .shared  // CPU/GPU/ANEが相互参照可能な最速モード
        heapDescriptor.type = .placement
        
        self.metalHeap = metalDevice.makeHeap(descriptor: heapDescriptor)
        
        // ANEが直接絵を焼き付けるキャンバスとなるMetalバッファを生成
        guard let heap = self.metalHeap else { return }
        self.displayBuffer = heap.makeBuffer(
            length: outputImageByteCount,
            options: .storageModeShared,
            offset: 0
        )
    }

    /// 外部のSwift/UIKit/Metalコードから毎フレームカメラ行列を更新するためのヘルパー
    func updateCamera(eye: simd_float3, target: simd_float3, up: simd_float3) {
        // 1. 3D数学：ビュー行列の計算
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
        
        // 2. レイトレーシングに必要な【逆行列】を算出 (CPU側で安全に処理)
        let invView = simdfMatrixInverse(viewMatrix)
        
        // 3. 🌟【64ch行優先アライメントハック】
        // 型安全な mutableView と withUnsafeMutablePointer を使って
        // ANEが期待する「行優先（Row-Major）」の順番に並び替えて直撃格納します。
        cameraMatrix64ChArray.mutableView(as: Float16.self).withUnsafeMutablePointer { pointer, _, _ in
            // 最初に64チャンネルすべてをゼロクリア
            for i in 0..<64 { pointer[i] = 0 }
            
            // 🌟【最重要ハック】
            // ANEの .view(4, 4) が「行」として正しく読み込めるように、
            // simd行列から[行、列]の組み合わせで横方向に16個並べ替えて詰め込みます。
            
            // --- 1行目 (Row 0) ---
            pointer[0]  = Float16(invView.columns.0.x) // [0,0]
            pointer[1]  = Float16(invView.columns.1.x) // [0,1]
            pointer[2]  = Float16(invView.columns.2.x) // [0,2]
            pointer[3]  = Float16(invView.columns.3.x) // [0,3]
            
            // --- 2行目 (Row 1) ---
            pointer[4]  = Float16(invView.columns.0.y) // [1,0]
            pointer[5]  = Float16(invView.columns.1.y) // [1,1]
            pointer[6]  = Float16(invView.columns.2.y) // [1,2]
            pointer[7]  = Float16(invView.columns.3.y) // [1,3]
            
            // --- 3行目 (Row 2) ---
            pointer[8]  = Float16(invView.columns.0.z) // [2,0]
            pointer[9]  = Float16(invView.columns.1.z) // [2,1]
            pointer[10] = Float16(invView.columns.2.z) // [2,2]
            pointer[11] = Float16(invView.columns.3.z) // [2,3]
            
            // --- 4行目 (Row 3) ---
            pointer[12] = Float16(invView.columns.0.w) // [3,0]
            pointer[13] = Float16(invView.columns.1.w) // [3,1]
            pointer[14] = Float16(invView.columns.2.w) // [3,2]
            pointer[15] = Float16(invView.columns.3.w) // [3,3]
        }
    }


    /// 1フレームレンダリング実行（Instrumentsで8msを記録した神のストリーム）
    func drawFrame() async throws {
        guard let raytracer = raytracerFunction,
              let canvasBuf = self.displayBuffer else { return }
        
        
        let inputs: [String: NDArray] = [
            "multiview_textures": multiviewTextureArray,
            "inv_view_matrix_64d": cameraMatrix64ChArray
        ]
        
        // 2. 🌟【ゼロコピー・ハック】
        // ANEの出力先として、確保してある Metal の MTLBuffer (canvasBuf) を直接指定。
        // これにより、ANEが推論を終えた瞬間にメインメモリへのコピーを挟むことなく、直接画面バッファへ絵が焼き付きます。
        nonisolated(unsafe) var outputViews = InferenceFunction.MutableViews()
        
        let shape: [Int] = [1, 1, 256, 256]
        let destinationView = NDArray.MutableRawView(
            metalBuffer: canvasBuf,
            byteOffset: 0,
            scalarType: .float16,
            shape: shape
        ).view(as: Float16.self)
        
        outputViews.insert(destinationView, for: "mul_138")
        
        // 3. ANE (NPU) のシリコンをダイレクト駆動！！
        let _ = try await raytracer.run(inputs: inputs, outputViews: outputViews)
    }
    
    /// simd_float4x4 の高速な逆行列計算用補助関数
    private func simdfMatrixInverse(_ m: matrix_float4x4) -> matrix_float4x4 {
        return m.inverse
    }
}
