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
    // Re Use buffer
    private var mvpWeights = [Float16](repeating: 0.0, count: 4 * 4 * 64)
    private var vertices = [Float16](repeating: 0.0, count: 1 * 4 * 3 * 64)
    private var colorsR = [Float16](repeating: 0.0, count: 64)
    private var colorsG = [Float16](repeating: 0.0, count: 64)
    private var colorsB = [Float16](repeating: 0.0, count: 64)
    
    private var debugTextureData: [Float16] = []
    init(){
        self.debugTextureData = geometry.createDebugCheckerboardTexture()
    }
    
    
    // Setup
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
        
            guard allowedExtensions.contains(url.pathExtension.lowercased()) else {
                print("Error: Invalid file extension for \(url.lastPathComponent)")
                return
            }
            _ = url.startAccessingSecurityScopedResource()
        }
        
        guard let pre = urls.first(where: { $0.lastPathComponent.lowercased().contains("pre") }),
              let rast = urls.first(where: { $0.lastPathComponent.lowercased().contains("rasterizer") || $0.lastPathComponent.lowercased().contains("render") }),
              let tex = urls.first(where: { $0.lastPathComponent.lowercased().contains("texture") }) else {
            print("Error: Could not identify all 3 models.")
            return
        }
        
        self.isLoading = true
        Task {
            self.mgDevice = await MGCreateSystemDefaultDevice(preURL: pre, rastURL: rast, texURL: tex)
            self.isLoading = false
            
            for url in urls { url.stopAccessingSecurityScopedResource() }
            
            if self.mgDevice != nil { self.startCameraRotation() }
        }
    }
    // Camera Loop
    func startCameraRotation() {
        timer?.invalidate()
        
        timer = Timer.scheduledTimer(withTimeInterval: 0.03, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self = self, let mgDevice = self.mgDevice, !self.isComputing else { return }
                
                self.isComputing = true
                self.angle += 0.05
                
                let radius: Float = 5.0
                let eyeX = radius * sin(self.angle)
                let eyeZ = radius * cos(self.angle)
                
                let cameraMatrix = mgDevice.createCameraMatrix(
                    eye: SIMD3<Float>(eyeX, 2.0, eyeZ),
                    target: SIMD3<Float>(0.0, 0.0, 0.0),
                    up: SIMD3<Float>(0.0, 1.0, 0.0)
                )

                // 配列をゼロリセット
                self.vertices = [Float16](repeating: 0.0, count: 1 * 64 * 4 * 3)
                self.mvpWeights = [Float16](repeating: 0.0, count: 1 * 64 * 4 * 4)

                let pyramidFaces: [[Float16]] = [
                    [ 0.0,  1.0, 0.0,  -1.0, -1.0, 1.0,   1.0, -1.0, 1.0,   0.0, 0.0, 0.0],
                    [ 0.0,  1.0, 0.0,   1.0, -1.0, 1.0,   1.0, -1.0, -1.0,  0.0, 0.0, 0.0],
                    [ 0.0,  1.0, 0.0,   1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  0.0, 0.0, 0.0],
                    [ 0.0,  1.0, 0.0,  -1.0, -1.0, -1.0, -1.0, -1.0, 1.0,   0.0, 0.0, 0.0]
                ]
                
                let faceColors: [(Float16, Float16, Float16)] = [
                    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)
                ]

                for i in 0..<4 {
                    self.colorsR[i] = faceColors[i].0
                    self.colorsG[i] = faceColors[i].1
                    self.colorsB[i] = faceColors[i].2
                    
                    for v in 0..<12 { self.vertices[(i * 12) + v] = pyramidFaces[i][v] }
                    for m in 0..<16 { self.mvpWeights[(i * 16) + m] = cameraMatrix[m] }
                }

                guard let mgCommandQueue = mgDevice.makeCommandQueue(),
                      let mgCommandBuffer = mgCommandQueue.makeCommandBuffer(),
                      let mgEncoder = mgCommandBuffer.makeRenderCommandEncoder() else {
                    self.isComputing = false
                    return
                }
                
                // 💡 API仕様に合わせて setVertexBytes でデータを渡す
                self.vertices.withUnsafeBytes { ptr in
                    mgEncoder.setVertexBytes(ptr.baseAddress!, length: self.vertices.count * 2, index: 0)
                }
                self.mvpWeights.withUnsafeBytes { ptr in
                    mgEncoder.setVertexBytes(ptr.baseAddress!, length: self.mvpWeights.count * 2, index: 1)
                }
                self.colorsR.withUnsafeBytes { ptr in
                    mgEncoder.setVertexBytes(ptr.baseAddress!, length: self.colorsR.count * 2, index: 2)
                }
                self.colorsG.withUnsafeBytes { ptr in
                    mgEncoder.setVertexBytes(ptr.baseAddress!, length: self.colorsG.count * 2, index: 3)
                }
                self.colorsB.withUnsafeBytes { ptr in
                    mgEncoder.setVertexBytes(ptr.baseAddress!, length: self.colorsB.count * 2, index: 4)
                }
                
                mgEncoder.setFragmentTexture(self.debugTextureData, index: 0)
                mgEncoder.drawPrimitives(vertexCount: 8)
                mgEncoder.endEncoding()
                
                do {
                    try await mgCommandBuffer.commit()
                } catch {
                    print("Inference error: \(error)")
                }
                
                self.isComputing = false
            }
        }
    }
    // Output
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
           
            for i in 0..<4 {
                if let buffer = mgDevice.getDisplayBuffer(index: i) {
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
