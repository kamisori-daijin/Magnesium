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
            print("Failed to find raytracer")
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
                self.update()
            }
        }
    }

    func update() {
        guard let mgDevice = self.mgDevice, !self.isComputing else { return }
        
        self.isComputing = true
        
        self.angle += 0.015
        let currentAngle = self.angle
        let halfAngle = currentAngle * 0.5
        
        let sinAngle = sin(currentAngle)
        let cosAngle = cos(currentAngle)
        let cosHalfAngle = cos(halfAngle)
        
        let radius: Float = 3.5
        let eyeX = radius * sinAngle
        let eyeY = radius * cosHalfAngle * 0.3 + 1.2
        let eyeZ = radius * cosAngle
        
        mgDevice.updateCamera(
            eye: SIMD3<Float>(eyeX, eyeY, eyeZ),
            target: SIMD3<Float>(0.0, 0.0, 0.0),
            up: SIMD3<Float>(0.0, 1.0, 0.0),
            time: currentAngle
        )

        guard let mgCommandQueue = self.mgCommandQueue,
              let mgCommandBuffer = mgCommandQueue.makeCommandBuffer() else {
            self.isComputing = false
            return
        }
        
        try? mgCommandBuffer.commit()
        
        self.currentEventValue += 1
        self.sharedEvent?.signaledValue = self.currentEventValue
        
        self.isComputing = false
    }

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

            // Get Current Buffer
            if let singleDisplayBuffer = mgDevice.getCurrentDisplayBuffer() {
                renderEncoder.setFragmentBuffer(singleDisplayBuffer, offset: 0, index: 0)
                renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }
            renderEncoder.endEncoding()
        }
        
        commandBuffer.present(drawable)
        
        commandBuffer.addCompletedHandler { _ in
            DispatchQueue.main.async { [weak self = self] in
                self?.update()
            }
        }
        
        commandBuffer.commit()
    }
}
