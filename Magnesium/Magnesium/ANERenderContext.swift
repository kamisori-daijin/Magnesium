
//
//  ANERenderContext.swift
//  Magnesium
//

import Foundation
import Metal
import MetalKit
import CoreAI
internal import UniformTypeIdentifiers
import simd

@MainActor
@Observable
class ANERenderContext {
    private var angle: Float = 0.0
    private var timer: Timer?
    private(set) var renderer: ANERenderer?
    private(set) var commandQueue: MTLCommandQueue?
    private var renderPipelineState: MTLRenderPipelineState?
    
    private var sharedEvent: MTLSharedEvent?
    private var currentEventValue: UInt64 = 0
    
    var isLoading = false
    var isComputing = false
    
    private let geometry = ANE3DGeometry()
    var activeDevice: MTLDevice?
    
    private var debugTextureData: [Float16] = []
    
    func setup(with device: MTLDevice) {
        self.activeDevice = device
        self.commandQueue = device.makeCommandQueue()
        self.sharedEvent = device.makeSharedEvent()
        
        // 元のテクスチャ初期化を100%維持
        self.debugTextureData = geometry.createDebugCheckerboardTexture()

        if let defaultLibrary = device.makeDefaultLibrary() {
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = defaultLibrary.makeFunction(name: "textureVertex")
            pipelineDescriptor.fragmentFunction = defaultLibrary.makeFunction(name: "textureFragment")
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
            pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
            pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .max
           
            self.renderPipelineState = try? device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        }
    }
    
    func openModelPicker() {
        guard let device = self.activeDevice else { return }
        let panel = NSOpenPanel()
       
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.item, .content, .data]
        panel.message = "Please select PreProcessor, Rasterizer, and Texture Processor .aimodel files."
        
        if let mainWindow = NSApplication.shared.windows.first(where: { $0.canBecomeKey }) {
            panel.beginSheetModal(for: mainWindow) { response in
                self.handlePanelResponse(response: response, panel: panel, device: device)
            }
        }
    }
    
    private func handlePanelResponse(response: NSApplication.ModalResponse, panel: NSOpenPanel, device: MTLDevice) {
        guard response == .OK, panel.urls.count == 3 else { return }
        
        let urls = panel.urls.map { $0.standardizedFileURL }
        guard let preProcessorURL = urls.first(where: { $0.lastPathComponent.lowercased().contains("pre") }),
              let rastURL = urls.first(where: { $0.lastPathComponent.lowercased().contains("rasterizer") || $0.lastPathComponent.lowercased().contains("render") }),
              let texURL = urls.first(where: { $0.lastPathComponent.lowercased().contains("texture") }) else {
            print("Error: Could not accurately identify all 3 models from filenames.")
            return
        }
        
        self.isLoading = true
        
        Task {
            defer { self.isLoading = false }
            do {
                // 💡 修正：1枚岩版 ANERenderer.swift の引数名に完全同期
                let loadedRenderer = try await ANERenderer(preURL: preProcessorURL, rastURL: rastURL, texURL: texURL, metalDevice: device)
                self.renderer = loadedRenderer
                print("All 3 models loaded successfully.")
                self.triggerSingleCompute()
                self.startCameraRotation()
            } catch {
                print("Failed to load models: \(error)")
            }
        }
    }

    func triggerSingleCompute() {
        guard let renderer = self.renderer, !isComputing else { return }
        
        self.isComputing = true
        renderer.updateTexture(pixelData: self.debugTextureData)
                
        Task { @MainActor in
            do {
                try await renderer.drawFrame()
                
                self.currentEventValue += 1
                self.sharedEvent?.signaledValue = self.currentEventValue
                
            } catch {
                print("Inference error: \(error)")
            }
    
            self.isComputing = false
        }
    }

    func startCameraRotation() {
        timer?.invalidate()
        
        timer = Timer.scheduledTimer(withTimeInterval: 0.03, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self = self, let renderer = self.renderer, !self.isComputing else { return }
                
                self.angle += 0.05
                
                let radius: Float = 5.5
                let eyeX = radius * sin(self.angle)
                let eyeZ = radius * cos(self.angle)
                
                let cameraMatrix = self.geometry.createCameraMatrix(
                    eye: SIMD3<Float>(eyeX, 5.0, eyeZ),
                    target: SIMD3<Float>(0.0, 0.0, 0.0),
                    up: SIMD3<Float>(0.0, 1.0, 0.0)
                )

                // 📐 新・前処理 Conv2d パイプラインへのコンバート処理
                // 4x4の simd_float4x4 を 1x1 Convの重み形状 [16] に綺麗に平坦化
                var mvpWeights = [Float16](repeating: 0.0, count: 16)
                for i in 0..<4 {
                    for j in 0..<4 {
                        mvpWeights[i * 4 + j] = cameraMatrix[j * 4 + i] }
                }

                // 64面分のフラットバッファの構築
                var vertices = [Float16](repeating: 0.0, count: 1 * 4 * 3 * 64)
                for faceIdx in 0..<64 {
                    vertices[3 * 64 * 3 + 0 * 64 + faceIdx] = 1.0 // p0_w
                    vertices[3 * 64 * 3 + 1 * 64 + faceIdx] = 1.0 // p1_w
                    vertices[3 * 64 * 3 + 2 * 64 + faceIdx] = 1.0 // p2_w
                }

                let pyramidFaces: [[[Float16]]] = [
                    [[ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, 1.0, 1.0]],
                    [[ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, -1.0, 1.0]],
                    [[ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, -1.0, 1.0]],
                    [[ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]],
                ]
                
                var colorsR = [Float16](repeating: 0.0, count: 64)
                var colorsG = [Float16](repeating: 0.0, count: 64)
                var colorsB = [Float16](repeating: 0.0, count: 64)
                
                let faceColors: [(Float16, Float16, Float16)] = [
                    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)
                ]
                
                for i in 0..<4 {
                    colorsR[i] = faceColors[i].0
                    colorsG[i] = faceColors[i].1
                    colorsB[i] = faceColors[i].2
                    
                    
                    // ANERenderContext.swift の startCameraRotation() 内
                    for i in 0..<4 {
                        colorsR[i] = faceColors[i].0
                        colorsG[i] = faceColors[i].1
                        colorsB[i] = faceColors[i].2
                        
                        // 各面の3つの頂点（v = 0, 1, 2）を、前処理が待つ正しい平面インデックスへ配置
                        for v in 0..<3 {
                            for ch in 0..<4 { // 0:X, 1:Y, 2:Z, 3:W
                                let channelOffset = ch * 3 * 64
                                let targetIndex = channelOffset + (v * 64) + i
                                
                                // [面i][頂点v][成分ch] から正確にデータを流し込む！
                                vertices[targetIndex] = pyramidFaces[i][v][ch]
                            }
                        }
                    }

                }

                
                // 💡 あなたの元の updateGeometry 引数名・構造のまま、5つの新しい配列を綺麗に受け渡す
                self.renderer?.updateGeometry(
                    vertices: vertices,
                    mvpWeights: mvpWeights,
                    r: colorsR,
                    g: colorsG,
                    b: colorsB
                )
                self.triggerSingleCompute()
            }
        }
    }
    
    func renderFrame(in view: MTKView) {
        view.colorPixelFormat = .bgra8Unorm
        
        guard let renderer = self.renderer,
              let queue = self.commandQueue,
              let pipeline = self.renderPipelineState,
              let sharedEvent = self.sharedEvent,
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable else { return }
        
        guard let commandBuffer = queue.makeCommandBuffer() else { return }
        
        if self.currentEventValue > 0 {
            commandBuffer.encodeWaitForEvent(sharedEvent, value: self.currentEventValue)
        }

        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
            renderEncoder.setRenderPipelineState(pipeline)
           
            // ✅ あなたの元の「displayBuffers」（複数形配列）の4回ループ描画を完全復活！！
            // 今回のラスタライザは先頭のバッファ（displayBuffers[0]）に全64面が一撃合成されます
            for i in 0..<4 {
                if let buffer = renderer.displayBuffers[i] {
                    renderEncoder.setFragmentBuffer(buffer, offset: 0, index: 0)
                    renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
                }
            }
            renderEncoder.endEncoding()
        }
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
