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
        
        if let defaultLibrary = device.makeDefaultLibrary() {
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = defaultLibrary.makeFunction(name: "textureVertex")
            pipelineDescriptor.fragmentFunction = defaultLibrary.makeFunction(name: "textureFragment")
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
           
            self.renderPipelineState = try? device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        }
    }
    
    func handleSelectedURLs(_ urls: [URL]) {

        guard let raytracerURL = urls.first(where: {
            $0.pathExtension.lowercased() == "aimodel" &&
            $0.lastPathComponent.lowercased().contains("raytracer")
        }) else {
            print("Faild to find raytracer")
            return
        }
        
        _ = raytracerURL.startAccessingSecurityScopedResource()
        
        self.isLoading = true
        Task {
          
            self.mgDevice = await MGCreateSystemDefaultDevice(raytracerURL: raytracerURL)
            self.isLoading = false
            
            raytracerURL.stopAccessingSecurityScopedResource()
            
            if self.mgDevice != nil {
                self.mgCommandQueue = self.mgDevice?.makeCommandQueue()
            }
        }
    }


    func update() async {
        guard let mgDevice = self.mgDevice, !self.isComputing else { return }
        
        self.isComputing = true
        // Camera angle
        self.angle += 0.015
        
        // 1. Camera Calucluration
        let radius: Float = 3.5
        let eyeX = radius * sin(self.angle)
        let eyeY = radius * cos(self.angle * 0.5) * 0.3 + 1.2
        let eyeZ = radius * cos(self.angle)
        
       
        mgDevice.updateCamera(
            eye: SIMD3<Float>(eyeX, eyeY, eyeZ),
            target: SIMD3<Float>(0.0, 0.0, 0.0),
            up: SIMD3<Float>(0.0, 1.0, 0.0)
        )

    
        mgDevice.withMultiviewTexturePointer { texturePointer in
            // 1 * 3 * 256 * 256
            for ch in 0..<3 {
                let chOffset = ch * 256 * 256
                
                for y in 0..<256 {
                    let yOffset = y * 256
                    // -1.0 〜 1.0
                    let normY = (Float(y) / 255.0) * 2.0 - 1.0
                    
                    for x in 0..<256 {
                        let normX = (Float(x) / 255.0) * 2.0 - 1.0
                        let index = chOffset + yOffset + x
                        
                        // 0.8（-0.4 〜 0.4）
                        let isInsideCube = (abs(normX) <= 0.4) && (abs(normY) <= 0.4)
                        
                        // Mask:1.0 another:black
                        texturePointer[index] = isInsideCube ? 1.0 : 0.0
                    }
                }
            }
        }
        

        guard let mgCommandQueue = self.mgCommandQueue,
              let mgCommandBuffer = mgCommandQueue.makeCommandBuffer() else {
            self.isComputing = false
            return
        }
        
        do {
          
            try await mgCommandBuffer.commit()
            
         
            self.currentEventValue += 1
            self.sharedEvent?.signaledValue = self.currentEventValue
        } catch {
            print("❌ ANE Inference Error: \(error)")
        }
        
        self.isComputing = false
    }

    /// Draw loop
    func renderFrame(in view: MTKView) {
        view.colorPixelFormat = .bgra8Unorm
        
        guard let mgDevice = self.mgDevice,
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

            if let singleDisplayBuffer = mgDevice.getDisplayBuffer() {
                renderEncoder.setFragmentBuffer(singleDisplayBuffer, offset: 0, index: 0)
                
         
                renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }
            renderEncoder.endEncoding()
        }
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
