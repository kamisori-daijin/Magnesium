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
    private(set) var commandQueue: MTLCommandQueue?
    private var renderPipelineState: MTLRenderPipelineState?
    
    private var sharedEvent: MTLSharedEvent?
    private var currentEventValue: UInt64 = 0
    
    var isLoading = false
    var isComputing = false
    
    var activeDevice: MTLDevice?
    
    private var debugTextureData: [Float16] = []
    
    // =================================================================
    // ⚙️ 初期セットアップ
    // =================================================================
    func setup(with device: MTLDevice) {
        self.activeDevice = device
        self.commandQueue = device.makeCommandQueue()
        self.sharedEvent = device.makeSharedEvent()
        
        self.debugTextureData = [Float16](repeating: 1.0, count: 640 * 400 * 3)

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
    // =================================================================
    // ⚔️ 3Dジオメトリパッキング ＆ カメラ回転ループ
    // =================================================================
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

                var mvpWeights = [Float16](repeating: 0.0, count: 4 * 4 * 64)
                var vertices = [Float16](repeating: 0.0, count: 1 * 4 * 3 * 64)
                var colorsR = [Float16](repeating: 0.0, count: 64)
                var colorsG = [Float16](repeating: 0.0, count: 64)
                var colorsB = [Float16](repeating: 0.0, count: 64)
                
                let wChannelOffset = 3 * 3 * 64
                for faceIdx in 0..<64 {
                    vertices[wChannelOffset + (0 * 64) + faceIdx] = 1.0
                    vertices[wChannelOffset + (1 * 64) + faceIdx] = 1.0
                    vertices[wChannelOffset + (2 * 64) + faceIdx] = 1.0
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
                    colorsR[slot] = faceColors[i].0; colorsG[slot] = faceColors[i].1; colorsB[slot] = faceColors[i].2
                    for v in 0..<3 {
                        for ch in 0..<4 {
                            let pIndex = (ch * 3 * 64) + (v * 64) + slot
                            var offsetValue = pyramidFaces[i][v][ch]
                            if ch == 0 { offsetValue -= 2.0 }
                            vertices[pIndex] = offsetValue
                        }
                    }
                    for m in 0..<16 { mvpWeights[m * 64 + slot] = cameraMatrix[m] }
                }

                for i in 0..<4 {
                    let slot = 4 + i
                    colorsR[slot] = faceColors[i].0; colorsG[slot] = faceColors[i].1; colorsB[slot] = faceColors[i].2
                    for v in 0..<3 {
                        for ch in 0..<4 {
                            let pIndex = (ch * 3 * 64) + (v * 64) + slot
                            var offsetValue = pyramidFaces[i][v][ch]
                            if ch == 0 { offsetValue += 2.0 }
                            vertices[pIndex] = offsetValue
                        }
                    }
                    for m in 0..<16 { mvpWeights[m * 64 + slot] = cameraMatrix[m] }
                }
                
                guard let mgCommandQueue = mgDevice.makeCommandQueue(),
                      let mgCommandBuffer = mgCommandQueue.makeCommandBuffer(),
                      let mgEncoder = mgCommandBuffer.makeRenderCommandEncoder() else {
                    self.isComputing = false
                    return
                }
                
                vertices.withUnsafeBytes { vertexPtr in
                    mgEncoder.setVertexBytes(vertexPtr.baseAddress!, length: vertices.count * 2, index: 0)
                }
                mvpWeights.withUnsafeBytes { mvpPtr in
                    mgEncoder.setVertexBytes(mvpPtr.baseAddress!, length: mvpWeights.count * 2, index: 1)
                }
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

    // =================================================================
    // 📺 最終画面出力（GPUレンダリング）
    // =================================================================
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
