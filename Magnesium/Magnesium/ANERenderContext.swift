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
    
    private let geometryEvaluator = ANETriangleGeometry(radius: 0.5, speed: 2.5)
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
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.item, .content, .data]
        panel.message = "Please select an .aimodel file."
        
        if let mainWindow = NSApplication.shared.windows.first(where: { $0.canBecomeKey }) {
            panel.beginSheetModal(for: mainWindow) { response in
                self.handlePanelResponse(response: response, panel: panel, device: device)
            }
        }
    }
    
    private func handlePanelResponse(response: NSApplication.ModalResponse, panel: NSOpenPanel, device: MTLDevice) {
        guard response == .OK, let selectedURL = panel.url else { return }
        let fixedURL = selectedURL.standardizedFileURL
        self.isLoading = true
        
        Task {
            defer {
                fixedURL.stopAccessingSecurityScopedResource()
                self.isLoading = false
            }
            do {
                let loadedRenderer = try await ANERenderer(modelURL: fixedURL, metalDevice: device)
                self.renderer = loadedRenderer
                print("Model loaded successfully.")
            } catch {
                print("Failed to load model: \(error)")
            }
        }
    }
    
    
    func triggerSingleCompute() {
        guard let renderer = self.renderer, !isComputing else { return }
        
        self.isComputing = true
        
        let currentTime = Float(Date().timeIntervalSince(startTime))
        let params = geometryEvaluator.calculatePackedParameters(forTime: currentTime)
        renderer.updateGeometryParameters(weights: params.weights, biases: params.biases)
        
        
        Task { @MainActor in
            do {
                print("Inference for a single frame")
                
                // Here, we wait completely until the ANE finishes writing its output to the MetalBuffer.
                try await renderer.drawFrame()
                
                print("Inference complete. Synchronous commit to Metal confirmed.")
            } catch {
                print("Inference error: \(error)")
            }
            // Ensure the completion flag is cleared.
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
        
        // Safely copy the ANE output to a Metal buffer here.
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
