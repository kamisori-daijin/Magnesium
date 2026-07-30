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
        

        self.debugTextureData = geometry.createDebugCheckerboardTexture()

        if let defaultLibrary = device.makeDefaultLibrary() {
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = defaultLibrary.makeFunction(name: "textureVertex")
            pipelineDescriptor.fragmentFunction = defaultLibrary.makeFunction(name: "textureFragment")
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
            pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .max
            pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .max
           
            self.renderPipelineState = try? device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        }
    }
    
    func openModelPicker() {
        guard let device = self.activeDevice else { return }
        let panel = NSOpenPanel()
        // 🌟 修正: MVP, Rasterizer, Texture の3つのファイルを同時に選択できるように変更
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.item, .content, .data]
        panel.message = "Please select MVP, Rasterizer, and Texture Processor .aimodel files."
        
        if let mainWindow = NSApplication.shared.windows.first(where: { $0.canBecomeKey }) {
            panel.beginSheetModal(for: mainWindow) { response in
                self.handlePanelResponse(response: response, panel: panel, device: device)
            }
        }
    }
    
    private func handlePanelResponse(response: NSApplication.ModalResponse, panel: NSOpenPanel, device: MTLDevice) {
        // 🌟 修正: ファイル選択の期待値を「3個」に変更
        guard response == .OK, panel.urls.count == 3 else { return }
        
        let urls = panel.urls.map { $0.standardizedFileURL }
        guard let mvpURL = urls.first(where: { $0.lastPathComponent.lowercased().contains("mvp") }),
              let rastURL = urls.first(where: { $0.lastPathComponent.lowercased().contains("rasterizer") || $0.lastPathComponent.lowercased().contains("render") }),
              let texURL = urls.first(where: { $0.lastPathComponent.lowercased().contains("texture") }) else {
            print("Error: Could not accurately identify all 3 models from filenames.")
            return
        }
        
        self.isLoading = true
        
        Task {
            defer { self.isLoading = false }
            do {
                // 🌟 修正: 拡張した3つのモデルを受け取るイニシャライザに差し替え
                let loadedRenderer = try await ANERenderer(mvpURL: mvpURL, rastURL: rastURL, texURL: texURL, metalDevice: device)
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
        
        // 🌟 追加: ANEラスタライザを回す直前に、生のRGBテクスチャバッファを確定コピー
        renderer.updateTexture(pixelData: self.debugTextureData)
                
        Task { @MainActor in
            do {
                try await withCheckedThrowingContinuation { continuation in
                    autoreleasepool {
                        Task {
                            do {
                                try await renderer.drawFrame()
                                continuation.resume()
                            } catch {
                                continuation.resume(throwing: error)
                            }
                        }
                    }
                }
                
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
                guard let self = self else { return }
                guard !self.isComputing else { return }
                
                self.angle += 0.05
                
                let radius: Float = 5.0
                let eyeX = radius * sin(self.angle)
                let eyeZ = radius * cos(self.angle)
                
                let cameraMatrix = self.geometry.createCameraMatrix(
                    eye: SIMD3<Float>(eyeX, 2.0, eyeZ),
                    target: SIMD3<Float>(0.0, 0.0, 0.0),
                    up: SIMD3<Float>(0.0, 1.0, 0.0)
                )
                
                let vertices = self.geometry.getPyramidVertices()
                // 🌟 追加: ANE3DGeometry+UV.swift からUV座標の配列をロード
                let uvs = self.geometry.getPyramidUVs()
                
                // 🌟 修正: 拡張した引数「uvs」を渡すようにシグネチャを修正
                self.renderer?.updateGeometry(vertices: vertices, cameraMatrix: cameraMatrix, uvs: uvs)
                
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
           
            var allBuffersReady = true
            
            for i in 0..<4 {
                if let buffer = renderer.displayBuffers[i] {
                    renderEncoder.setFragmentBuffer(buffer, offset: 0, index: i)
                } else {
                    allBuffersReady = false
                }
            }
            
            if allBuffersReady {
                renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }
            
            renderEncoder.endEncoding()
        }
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
