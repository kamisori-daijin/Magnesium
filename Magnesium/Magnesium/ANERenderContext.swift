//
//  ANERenderContext.swift
//  Magnesium
//

import Foundation
import Metal
import MetalKit
import MagnesiumKit

@MainActor
@Observable
class ANERenderContext {
    private var angle: Float = 0.0
    private var timer: Timer?
    private(set) var mgDevice: MGDevice?
    private let geometry = ANE3DGeometry()
    private(set) var commandQueue: MTLCommandQueue?
    private var renderPipelineState: MTLRenderPipelineState?
    
    private var sharedEvent: MTLSharedEvent?
    private var currentEventValue: UInt64 = 0
    
    var isLoading = false
    var isComputing = false
    
    var activeDevice: MTLDevice?
    private var mvpWeights = [Float16](repeating: 0.0, count: 4 * 4 * 64)
    private var vertices = [Float16](repeating: 0.0, count: 1 * 4 * 3 * 64)
    private var colorsR = [Float16](repeating: 0.0, count: 64)
    private var colorsG = [Float16](repeating: 0.0, count: 64)
    private var colorsB = [Float16](repeating: 0.0, count: 64)
    
    private var debugTextureData: [Float16] = []
    
    init(){
        // 128x128 Texture
        self.debugTextureData = geometry.createDebugCheckerboardTexture()
    }
    
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
            pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
            pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .max
           
            self.renderPipelineState = try? device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        }
    }
    
    func handleSelectedURLs(_ urls: [URL]) {
        guard urls.count == 3 else { return }
        let allowedExtensions = ["aimodel"]
        
        for url in urls {
            guard allowedExtensions.contains(url.pathExtension.lowercased()) else { return }
            _ = url.startAccessingSecurityScopedResource()
        }
        
        guard let pre = urls.first(where: { $0.lastPathComponent.lowercased().contains("pre") }),
              let rast = urls.first(where: { $0.lastPathComponent.lowercased().contains("rasterizer") || $0.lastPathComponent.lowercased().contains("render") }),
              let tex = urls.first(where: { $0.lastPathComponent.lowercased().contains("texture") }) else { return }
        
        self.isLoading = true
        Task {
            self.mgDevice = await MGCreateSystemDefaultDevice(preURL: pre, rastURL: rast, texURL: tex)
            self.isLoading = false
            for url in urls { url.stopAccessingSecurityScopedResource() }
            if self.mgDevice != nil { self.startCameraRotation() }
        }
    }

    func startCameraRotation() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.03, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self = self, let mgDevice = self.mgDevice, !self.isComputing else { return }
                self.isComputing = true
                self.angle += 0.05
                
                let radius: Float = 5.5
                let eyeX = radius * sin(self.angle)
                let eyeZ = radius * cos(self.angle)
                
                let cameraMatrix = mgDevice.createCameraMatrix(
                    eye: SIMD3<Float>(eyeX, 5.0, eyeZ),
                    target: SIMD3<Float>(0.0, 0.0, 0.0),
                    up: SIMD3<Float>(0.0, 1.0, 0.0)
                )

                let wChannelOffset = 3 * 3 * 64
                for faceIdx in 0..<64 {
                    self.vertices[wChannelOffset + (0 * 64) + faceIdx] = 1.0
                    self.vertices[wChannelOffset + (1 * 64) + faceIdx] = 1.0
                    self.vertices[wChannelOffset + (2 * 64) + faceIdx] = 1.0
                }

                let pyramidFaces: [[[Float16]]] = [
                    [[ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, 1.0, 1.0]],
                    [[ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, 1.0, 1.0], [ 1.0, -1.0, -1.0, 1.0]],
                    [[ 0.0,  1.0, 0.0, 1.0], [ 1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, -1.0, 1.0]],
                    [[ 0.0,  1.0, 0.0, 1.0], [-1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]],
                ]
                
                let faceColors: [(Float16, Float16, Float16)] = [
                    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)
                ]

                for i in 0..<4 {
                    let slot = i
                    self.colorsR[slot] = faceColors[i].0; self.colorsG[slot] = faceColors[i].1; self.colorsB[slot] = faceColors[i].2
                    for v in 0..<3 {
                        for ch in 0..<4 {
                            let pIndex = (ch * 3 * 64) + (v * 64) + slot
                            var offsetValue = pyramidFaces[i][v][ch]
                            if ch == 0 { offsetValue -= 2.0 }
                            self.vertices[pIndex] = offsetValue
                        }
                    }
                    for m in 0..<16 { self.mvpWeights[m * 64 + slot] = cameraMatrix[m] }
                }
                
                guard let mgCommandQueue = mgDevice.makeCommandQueue(),
                      let mgCommandBuffer = mgCommandQueue.makeCommandBuffer(),
                      let mgEncoder = mgCommandBuffer.makeRenderCommandEncoder() else {
                    self.isComputing = false
                    return
                }
                
                self.vertices.withUnsafeBytes { mgEncoder.setVertexBytes($0.baseAddress!, length: self.vertices.count * 2, index: 0) }
                self.mvpWeights.withUnsafeBytes { mgEncoder.setVertexBytes($0.baseAddress!, length: self.mvpWeights.count * 2, index: 1) }
                self.colorsR.withUnsafeBytes { mgEncoder.setVertexBytes($0.baseAddress!, length: self.colorsR.count * 2, index: 2) }
                self.colorsG.withUnsafeBytes { mgEncoder.setVertexBytes($0.baseAddress!, length: self.colorsG.count * 2, index: 3) }
                self.colorsB.withUnsafeBytes { mgEncoder.setVertexBytes($0.baseAddress!, length: self.colorsB.count * 2, index: 4) }
                
                mgEncoder.setFragmentTexture(self.debugTextureData, index: 0)
                mgEncoder.drawPrimitives(vertexCount: 8)
                mgEncoder.endEncoding()
                
                do {
                    try await mgCommandBuffer.commit()
                    self.currentEventValue += 1
                    self.sharedEvent?.signaledValue = self.currentEventValue
                } catch {
                    print("Inference error: \(error)")
                }
                
                self.isComputing = false
            }
        }
    }

    func renderFrame(in view: MTKView) {
        view.colorPixelFormat = .bgra8Unorm
        guard let mgDevice = self.mgDevice, let queue = self.commandQueue, let pipeline = self.renderPipelineState,
              let sharedEvent = self.sharedEvent, let renderPassDescriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable else { return }
        
        guard let commandBuffer = queue.makeCommandBuffer() else { return }
        if self.currentEventValue > 0 {
            commandBuffer.encodeWaitForEvent(sharedEvent, value: self.currentEventValue)
        }

        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
            renderEncoder.setRenderPipelineState(pipeline)
           
            for tileIndex in 0..<135 {
                if let buffer = mgDevice.getDisplayBuffer(index: 0) {
                    var index = tileIndex
                    renderEncoder.setVertexBytes(&index, length: MemoryLayout<Int>.size, index: 1)
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
