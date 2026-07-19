//
//  ANERenderContext.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/07/12.
//

import Foundation
import Metal
import MetalKit
import CoreAI
internal import UniformTypeIdentifiers

@MainActor
@Observable
class ANERenderContext {
    private(set) var renderer: ANERenderer?
    private(set) var commandQueue: MTLCommandQueue?
    private var renderPipelineState: MTLRenderPipelineState?
    
    var isLoading = false
    var isShowingPicker = false
    var isComputing = false
    
    private let geometry = ANE3DGeometry()
    private let startTime = Date()
    var activeDevice: MTLDevice?
    
    func setup(with device: MTLDevice) {
        self.activeDevice = device
        self.commandQueue = device.makeCommandQueue()
        
        if let defaultLibrary = device.makeDefaultLibrary() {
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = defaultLibrary.makeFunction(name: "textureVertex")
            pipelineDescriptor.fragmentFunction = defaultLibrary.makeFunction(name: "textureFragment")
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            self.renderPipelineState = try? device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        }
    }
    
    func openModelPicker() {
        guard let device = self.activeDevice else { return }
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true // 2つのファイルを選択可能に
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
        
        // ファイル名からMVPとRasterizerを判別
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
            } catch {
                print("Failed to load models: \(error)")
            }
        }
    }
    
    func triggerSingleCompute() {
        guard let renderer = self.renderer, !isComputing else { return }
        
        self.isComputing = true
        
        // カメラを回転させるなどのアニメーション処理をここに追加可能
        let cameraMatrix = geometry.createCameraMatrix(
            eye: SIMD3<Float>(2.0, 2.0, -5.0),
            target: SIMD3<Float>(0.0, 0.0, 0.0),
            up: SIMD3<Float>(0.0, 1.0, 0.0)
        )
        let vertices = geometry.getPyramidVertices()
        
        renderer.updateGeometry(vertices: vertices, cameraMatrix: cameraMatrix)
        
        Task { @MainActor in
            do {
                print("Inference for a single frame")
                try await renderer.drawFrame()
                print("Inference complete.")
            } catch {
                print("Inference error: \(error)")
            }
            self.isComputing = false
        }
    }
    
    func renderFrame(in view: MTKView) {
        view.colorPixelFormat = .bgra8Unorm
        
        guard let renderer = self.renderer,
              let queue = self.commandQueue,
              let pipeline = self.renderPipelineState,
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable else { return }
        
        if let displayBuffer = renderer.displayBuffer {
            renderer.updateDisplayBuffer(displayBuffer)
        }
        
        guard let commandBuffer = queue.makeCommandBuffer() else { return }
        
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
            renderEncoder.setRenderPipelineState(pipeline)
            
            if let displayBuffer = renderer.displayBuffer {
                renderEncoder.setFragmentBuffer(displayBuffer, offset: 0, index: 0)
            }
            
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            renderEncoder.endEncoding()
        }
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
