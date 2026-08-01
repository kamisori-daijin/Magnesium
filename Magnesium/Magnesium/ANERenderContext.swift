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
    
    private var renderTask: Task<Void, Never>?
    
    private(set) var renderer: ANERenderer?
    private(set) var commandQueue: MTLCommandQueue?
    private var renderPipelineState: MTLRenderPipelineState?
    
    private var sharedEvent: MTLSharedEvent?
    private var sharedEventListener: MTLSharedEventListener?
 
    private var cpuSignaledValue: UInt64 = 0
    private var gpuPresentedValue: UInt64 = 0
    
    var isLoading = false
    var isComputing = false
    
    private let geometry = ANE3DGeometry()
    var activeDevice: MTLDevice?
    private var debugTextureData: [Float16] = []
    
    func setup(with device: MTLDevice) {
        self.activeDevice = device
        self.commandQueue = device.makeCommandQueue()
        self.sharedEvent = device.makeSharedEvent()
        self.sharedEventListener = MTLSharedEventListener()
        
        // Load Texture
        self.debugTextureData = geometry.createDebugCheckerboardTexture()

        if let defaultLibrary = device.makeDefaultLibrary() {
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = defaultLibrary.makeFunction(name: "textureVertex")
            pipelineDescriptor.fragmentFunction = defaultLibrary.makeFunction(name: "textureFragment")
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
            pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
            pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .max
           
            do {
                self.renderPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
                print("✅ Metal Pipeline State initialized successfully!")
            } catch {
                print("❌ Failed to create render pipeline state: \(error)")
            }
        }
    }
    
    func openModelPicker() {
        guard let device = self.activeDevice else { return }
        let panel = NSOpenPanel()
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
        guard response == .OK, panel.urls.count == 3 else { return }
        
        let urls = panel.urls.map { $0.standardizedFileURL }
        guard let mvpURL = urls.first(where: { $0.lastPathComponent.lowercased().contains("mvp") }),
              let rastURL = urls.first(where: { $0.lastPathComponent.lowercased().contains("rasterizer") }),
              let texURL = urls.first(where: { $0.lastPathComponent.lowercased().contains("texture") }) else {
            print("❌ Model identification failed.")
            return
        }

        self.isLoading = true
        
        Task {
            defer { self.isLoading = false }
            do {
                let loadedRenderer = try await ANERenderer(mvpURL: mvpURL, rastURL: rastURL, texURL: texURL, metalDevice: device)
                self.renderer = loadedRenderer
                print("✅ All 3 models loaded successfully.")
                
                self.startMainRenderLoop()
            } catch {
                print("❌ Failed to load models: \(error)")
            }
        }
    }

    func startMainRenderLoop() {
        renderTask?.cancel()
        
        renderTask = Task { @MainActor in
            while !Task.isCancelled {
                guard let renderer = self.renderer, let sharedEvent = self.sharedEvent else { break }
                
                if cpuSignaledValue > gpuPresentedValue {
                    guard let listener = self.sharedEventListener else { break }
                    
                    await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                        sharedEvent.notify(listener, atValue: cpuSignaledValue) { _, _ in
                            continuation.resume()
                        }
                    }
                    gpuPresentedValue = cpuSignaledValue
                }
                
                // 2. Rotate Camera
                self.angle += 0.05
                let radius: Float = 2.5
                let eyeX = radius * sin(self.angle)
                let eyeZ = radius * cos(self.angle)
                
                let cameraMatrix = self.geometry.createCameraMatrix(
                    eye: SIMD3<Float>(eyeX, 1.2, eyeZ),
                    target: SIMD3<Float>(0.0, 0.0, 0.0),
                    up: SIMD3<Float>(0.0, 1.0, 0.0)
                )
                
                let vertices = self.geometry.getPyramidVertices()
                let uvs = self.geometry.getPyramidUVs()
                renderer.updateGeometry(vertices: vertices, cameraMatrix: cameraMatrix, uvs: uvs)
                
                // 3. Copy
                renderer.updateTexture(pixelData: self.debugTextureData)
                
                // 4. Inference and Sync
                self.isComputing = true
                do {
                    try await renderer.drawFrame()
                    
                 
                    self.cpuSignaledValue += 1
                } catch {
                    print("Inference error: \(error)")
                }
                self.isComputing = false
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
        
  
        if self.cpuSignaledValue > 0 {
            commandBuffer.encodeWaitForEvent(sharedEvent, value: self.cpuSignaledValue)
        }

        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
            renderEncoder.setRenderPipelineState(pipeline)
           
            // Draw
            for i in 0..<4 {
                if let buffer = renderer.displayBuffers[i] {
                    renderEncoder.setFragmentBuffer(buffer, offset: 0, index: 0)
                    renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
                }
            }
            
            renderEncoder.endEncoding()
        }
        
        // Commit
        commandBuffer.present(drawable)
        
   
        commandBuffer.encodeSignalEvent(sharedEvent, value: self.cpuSignaledValue)
        
        commandBuffer.commit()
    }
    
    isolated deinit {
        renderTask?.cancel()
    }
}
