//
//  ANERenderContext.swift
//  Magnesium
//

import Foundation
import Metal
import MetalKit
import MagnesiumKit
import Observation

@MainActor
@Observable
class ANERenderContext {
    private var angle: Float = 0.0
    private(set) var mgDevice: MGDevice?
    private var commandQueue: MTLCommandQueue?
    private var mgCommandQueue: MGCommandQueue?
    private var renderPipelineState: MTLRenderPipelineState?
    
    private var sharedEvent: MTLSharedEvent?
    private var currentEventValue: UInt64 = 0
    
    var isLoading = false
    var isComputing = false
    
    var activeDevice: MTLDevice?
    
    init() {}
    
    func setup(with device: MTLDevice) {
        self.activeDevice = device
        self.commandQueue = device.makeCommandQueue()
        self.sharedEvent = device.makeSharedEvent()
        
        // 🌟 修正：ANEが直接 Float16 で焼き付けた 256x256 の結果画像を画面全体に転送するシェーダーを設定
        if let defaultLibrary = device.makeDefaultLibrary() {
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = defaultLibrary.makeFunction(name: "textureVertex")
            pipelineDescriptor.fragmentFunction = defaultLibrary.makeFunction(name: "textureFragment")
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
           
            self.renderPipelineState = try? device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        }
    }
    
    /// 🌟 修正：3ファイル待受から、新レイトレーサー 1ファイル（.aimodel）待受へとハック
    func handleSelectedURLs(_ urls: [URL]) {
        // カメラ＆自動法線対応の新神回路のURLをピンポイント抽出
        guard let raytracerURL = urls.first(where: {
            $0.pathExtension.lowercased() == "aimodel" &&
            $0.lastPathComponent.lowercased().contains("raytracer")
        }) else {
            print("❌ Error: 有効な raytracer.aimodel が見つかりません。")
            return
        }
        
        _ = raytracerURL.startAccessingSecurityScopedResource()
        
        self.isLoading = true
        Task {
            // 🌟 修正：1つのURLでMagnesiumDeviceを起動
            self.mgDevice = await MGCreateSystemDefaultDevice(raytracerURL: raytracerURL)
            self.isLoading = false
            
            raytracerURL.stopAccessingSecurityScopedResource()
            
            if self.mgDevice != nil {
                self.mgCommandQueue = self.mgDevice?.makeCommandQueue()
            }
        }
    }

    /// 🌟 毎フレームの定期更新ループ（Instrumentsで8msを叩き出したストリームの心臓部）
    func update() async {
        guard let mgDevice = self.mgDevice, !self.isComputing else { return }
        
        self.isComputing = true
        // 毎フレームカメラの回転角度を進める
        self.angle += 0.015
        
        // 1. 🌟【3Dカメラ制御】ワールド空間でのカメラ旋回軌道計算
        let radius: Float = 3.5
        let eyeX = radius * sin(self.angle)
        let eyeY = radius * cos(self.angle * 0.5) * 0.3 + 1.2 // 上下にもほんのり揺らす
        let eyeZ = radius * cos(self.angle)
        
        // 新しいMGDeviceのインターフェースを叩き、64ch行優先アライメントハック行列を自動構築
        mgDevice.updateCamera(
            eye: SIMD3<Float>(eyeX, eyeY, eyeZ),
            target: SIMD3<Float>(0.0, 0.0, 0.0),
            up: SIMD3<Float>(0.0, 1.0, 0.0)
        )

        // 2. 🌟【安全バインドブリッジ】3面図テクスチャ（1x3x256x256）へ立方体を彫刻
        // トーラス頂点や無駄なループを完全全廃し、ポインタ直撃で3面図マスクを一撃生成
        mgDevice.withMultiviewTexturePointer { texturePointer in
            // ポインタ全体（1 * 3 * 256 * 256 要素）を一巡
            for ch in 0..<3 {
                let chOffset = ch * 256 * 256
                
                for y in 0..<256 {
                    let yOffset = y * 256
                    // -1.0 〜 1.0 の正規化空間をシミュレート
                    let normY = (Float(y) / 255.0) * 2.0 - 1.0
                    
                    for x in 0..<256 {
                        let normX = (Float(x) / 255.0) * 2.0 - 1.0
                        let index = chOffset + yOffset + x
                        
                        // 一辺の長さが 0.8（-0.4 〜 0.4）の立方体（正方形）マスクを3面に焼き付ける
                        let isInsideCube = (abs(normX) <= 0.4) && (abs(normY) <= 0.4)
                        
                        // マスク領域なら 1.0 (白)、外なら 0.0 (黒)
                        texturePointer[index] = isInsideCube ? 1.0 : 0.0
                    }
                }
            }
        }
        
        // 3. 🌟【一本道エンコーディング】
        guard let mgCommandQueue = self.mgCommandQueue,
              let mgCommandBuffer = mgCommandQueue.makeCommandBuffer() else {
            self.isComputing = false
            return
        }
        
        do {
            // ANEのNPUコアを直撃駆動（8.02msの超高速レイトレーシング実行）
            try await mgCommandBuffer.commit()
            
            // レンダリング完了をMetal（GPU側）に通知するためのセマフォ同期イベントをインクリメント
            self.currentEventValue += 1
            self.sharedEvent?.signaledValue = self.currentEventValue
        } catch {
            print("❌ ANE Inference Error: \(error)")
        }
        
        self.isComputing = false
    }

    /// 🌟 Metal（GPU側）の画面リフレッシュ描画ループ
    func renderFrame(in view: MTKView) {
        view.colorPixelFormat = .bgra8Unorm
        
        guard let mgDevice = self.mgDevice,
              let queue = self.commandQueue,
              let pipeline = self.renderPipelineState,
              let sharedEvent = self.sharedEvent,
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable else { return }
        
        guard let commandBuffer = queue.makeCommandBuffer() else { return }
        
        // 🌟【最重要イベント同期】ANE（NPU）が画像バッファを焼き付け終えるまで、GPU側の実行を一時停止（ウェイト）させる
        if self.currentEventValue > 0 {
            commandBuffer.encodeWaitForEvent(sharedEvent, value: self.currentEventValue)
        }

        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
            renderEncoder.setRenderPipelineState(pipeline)
           
            // 🌟 修正：4枚バラバラの旧バッファを廃止し、ANEがゼロコピー直撃した
            // 単一の Float16 画素バッファ（256x256）をGPUのインデックス0番にバインド！
            if let singleDisplayBuffer = mgDevice.getDisplayBuffer() {
                renderEncoder.setFragmentBuffer(singleDisplayBuffer, offset: 0, index: 0)
                
                // 画面全体（テクスチャ展開用ポリゴン）に一撃で描画命令を発行
                renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }
            renderEncoder.endEncoding()
        }
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
