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
    private(set) var commandQueue: MTLCommandQueue?
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
            pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
            pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .max
           
            self.renderPipelineState = try? device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        }
    }
    
    func handleSelectedURLs(_ urls: [URL]) {
        guard urls.count == 1, let modelURL = urls.first else { return }
        guard modelURL.pathExtension.lowercased() == "aimodel" else { return }
        
        _ = modelURL.startAccessingSecurityScopedResource()
        
        self.isLoading = true
        Task {
            self.mgDevice = await MGCreateSystemDefaultDevice(modelURL: modelURL)
            self.isLoading = false
            
            modelURL.stopAccessingSecurityScopedResource()
            
            if self.mgDevice != nil {
                self.mgCommandQueue = self.mgDevice?.makeCommandQueue()
                self.update() // First run
            }
        }
    }

    func update() {
        guard let mgDevice = self.mgDevice, !self.isComputing else { return }
        
        self.isComputing = true
        self.angle += 0.05
        
        let radius: Float = 6.0
        let eyeX = radius * sin(self.angle)
        let eyeZ = radius * cos(self.angle)
        
        let cameraMatrix = mgDevice.createCameraMatrix(
            eye: SIMD3<Float>(eyeX, 4.0, eyeZ),
            target: SIMD3<Float>(0.0, 0.0, 0.0),
            up: SIMD3<Float>(0.0, 1.0, 0.0)
        )

        mgDevice.withGeometryPointers { vertices, mvpWeights, normals, lightDir, colorsR, colorsG, colorsB in
            
            // ライトの方向を設定 (例: 斜め上からの光)
            lightDir[0] = Float16(0.0) // X
            lightDir[1] = Float16(1.0) // Y
            lightDir[2] = Float16(0.0) // Z
            // 残りの61チャンネルは0のままでOK
            
            for faceIdx in 0..<64 {
                for v in 0..<3 {
                    let wIndex = (faceIdx * 4 * 3) + (3 * 3) + v
                    vertices[wIndex] = 1.0
                }
            }

            let faces = TorusGeometry.generateFaces()

            for slot in 0..<min(faces.count, 64) {
                let face = faces[slot]
                colorsR[slot] = Float16(slot % 3 == 0 ? 1.0 : 0.0)
                colorsG[slot] = Float16(slot % 3 == 1 ? 1.0 : 0.0)
                colorsB[slot] = Float16(slot % 3 == 2 ? 1.0 : 0.0)
                
                for ch in 0..<4 {
                    for v in 0..<3 {
                        let pIndex = (slot * 4 * 3) + (ch * 3) + v
                        vertices[pIndex] = face[v][ch]
                    }
                }
                
                // 法線のダミーデータ設定（実際にはジオメトリから計算した法線を入れます）
                for v in 0..<3 {
                    let nIndex = (slot * 3 * 3) + (v * 3)
                    normals[nIndex + 0] = Float16(0.0)
                    normals[nIndex + 1] = Float16(1.0) // 上向きの法線
                    normals[nIndex + 2] = Float16(0.0)
                }
                
                for i in 0..<4 {
                    for j in 0..<4 {
                        let mIndex = (slot * 4 * 4) + (i * 4) + j
                        mvpWeights[mIndex] = cameraMatrix[i * 4 + j]
                    }
                }
            }
        }
        
        guard let mgCommandQueue = self.mgCommandQueue,
              let mgCommandBuffer = mgCommandQueue.makeCommandBuffer(),
              let mgEncoder = mgCommandBuffer.makeRenderCommandEncoder() else {
            self.isComputing = false
            return
        }
        
        mgEncoder.withFragmentTexturePointer(index: 0) { texturePointer in
            for y in 0..<256 {
                for x in 0..<256 {
                    let index = (y * 256 + x) * 3
                    let u = Float16(x) / 255.0
                    let v = Float16(y) / 255.0
                    texturePointer[index + 0] = u
                    texturePointer[index + 1] = v
                    texturePointer[index + 2] = 1.0 - u
                }
            }
        }
        
        mgEncoder.endEncoding()
        
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
