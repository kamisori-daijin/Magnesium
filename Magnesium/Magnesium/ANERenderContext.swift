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
    
    func setup(with device: MTLDevice) {
        self.activeDevice = device
        self.commandQueue = device.makeCommandQueue()
        self.sharedEvent = device.makeSharedEvent()

        
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
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.item, .content, .data]
        panel.message = "Please select both MVP and Rasterizer .aimodel files."
        
        if let mainWindow = NSApplication.shared.windows.first(where: { $0.canBecomeKey }) {
            panel.beginSheetModal(for: mainWindow) { response in
                self.handlePanelResponse(response: response, panel: panel, device: device)
            }
        }
    }
    
    private func handlePanelResponse(response: NSApplication.ModalResponse, panel: NSOpenPanel, device: MTLDevice) {
        guard response == .OK, panel.urls.count == 2 else { return }
        
        let urls = panel.urls.map { $0.standardizedFileURL }
        guard let mvpURL = urls.first(where: { $0.lastPathComponent.contains("mvp") }),
              let rastURL = urls.first(where: { $0.lastPathComponent.contains("rasterizer") }) else {
            print("Error: Could not identify MVP and Rasterizer models.")
            return
        }
        
        self.isLoading = true
        
        Task {
            defer { self.isLoading = false }
            do {
                let loadedRenderer = try await ANERenderer(mvpURL: mvpURL, rastURL: rastURL, metalDevice: device)
                self.renderer = loadedRenderer
                print("Models loaded successfully.")
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
        
        // 👇 ここにあった cameraMatrix の生成と updateGeometry を削除しました
        
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
            // Task を使って MainActor のコンテキストで実行する
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                
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
                self.renderer?.updateGeometry(vertices: vertices, cameraMatrix: cameraMatrix)
                
                self.triggerSingleCompute()
            }
        }
    }
    
    func renderFrame(in view: MTKView) {
        view.colorPixelFormat = .bgra8Unorm
        
        // displayBuffer が nil の場合は描画をスキップしてエラーを防ぐ
        guard let renderer = self.renderer,
              //let displayBuffer = renderer.displayBuffers,
              let queue = self.commandQueue,
              let pipeline = self.renderPipelineState,
              let sharedEvent = self.sharedEvent,
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable else { return }
        
        guard let commandBuffer = queue.makeCommandBuffer() else { return }
        
        if self.currentEventValue > 0 {
            commandBuffer.encodeWaitForEvent(sharedEvent, value: self.currentEventValue)
        }
        //print("🎨 Rendering Frame - Event: \(self.currentEventValue), Buffer: \(displayBuffer.length) bytes")
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
            renderEncoder.setRenderPipelineState(pipeline)
           
            var allBuffersReady = true
            
            // 4つのバッファをチェックしながらセット
            for i in 0..<4 {
                if let buffer = renderer.displayBuffers[i] {
                    renderEncoder.setFragmentBuffer(buffer, offset: 0, index: i)
                } else {
                    // 1つでもnilがあればフラグを折る
                    allBuffersReady = false
                }
            }
            
            // すべてのバッファが揃っている時だけ描画する
            if allBuffersReady {
                renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }
            
            renderEncoder.endEncoding()
        }
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
