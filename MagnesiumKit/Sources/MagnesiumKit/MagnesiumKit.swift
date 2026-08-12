// Public API Bridge

import Foundation
import CoreAI
import Metal

// Public API

public protocol MGDevice: AnyObject {
    var name: String { get }
    func makeCommandQueue() -> MGCommandQueue?
}

public protocol MGCommandQueue: AnyObject {
    func makeCommandBuffer() -> MGCommandBuffer?
}

public protocol MGCommandBuffer: AnyObject {
    func makeRenderCommandEncoder() -> MGRenderCommandEncoder?
    func commit()
}

public protocol MGRenderCommandEncoder: AnyObject {
    func setVertexBytes(_ bytes: UnsafeRawPointer, length: Int, index: Int)
    func setFragmentTexture(_ texture: [Float16], index: Int)
    func drawPrimitives(vertexCount: Int)
    func endEncoding()
}



internal final class MagnesiumDevice: MGDevice {
    public let name = "MagnesiumKit"
    
   internal var renderer: ANERenderer?
    
    public init() async {
        do {
            if let texURL = Bundle.main.url(forResource: "ane_texture_processor", withExtension: "aimodel"),
               let rstURL = Bundle.main.url(forResource: "ane_3d_rasterizer", withExtension: "aimodel"),
               let preURL = Bundle.main.url(forResource: "ane_pre_processor", withExtension: "aimodel") {
                
                guard let systemMetalDevice = MTLCreateSystemDefaultDevice() else {
                    print("Error: Failed to create default Metal device.")
                    return
                }
                
                self.renderer = try await ANERenderer(
                    preURL: preURL,
                    rastURL: rstURL,
                    texURL: texURL,
                    metalDevice: systemMetalDevice
                )
                print("Success: 3 Models and Metal API fully bound.")
            } else {
                print("Error: Model URLs not found in module bundle.")
            }
        } catch {
            print("Error: Failed to load models: \(error)")
        }
    }
    
    public func makeCommandQueue() -> MGCommandQueue? {
        return MagnesiumCommandQueue(device: self)
    }
}

private final class MagnesiumCommandQueue: MGCommandQueue {
    let device: MagnesiumDevice
    init(device: MagnesiumDevice) { self.device = device }
    
    func makeCommandBuffer() -> MGCommandBuffer? {
        return MagnesiumCommandBuffer(device: device)
    }
}

private final class MagnesiumCommandBuffer: MGCommandBuffer {
    let device: MagnesiumDevice
    private var encoder: MagnesiumRenderCommandEncoder?
    
    init(device: MagnesiumDevice) { self.device = device }
    
    func makeRenderCommandEncoder() -> MGRenderCommandEncoder? {
        let enc = MagnesiumRenderCommandEncoder(device: device)
        self.encoder = enc
        return enc
    }
    
    func commit() {
        guard let enc = encoder, let renderer = device.renderer else { return }
        print("Commit Success")
    }
}

private final class MagnesiumRenderCommandEncoder: MGRenderCommandEncoder {
    let device: MagnesiumDevice
    var boundVertices: [Float16] = []
    var boundTexture: [Float16] = []
    
    init(device: MagnesiumDevice) { self.device = device }
    
    func setVertexBytes(_ bytes: UnsafeRawPointer, length: Int, index: Int) {
    
        let count = length / MemoryLayout<Float16>.size
        let ptr = bytes.bindMemory(to: Float16.self, capacity: count)
        self.boundVertices = Array(UnsafeBufferPointer(start: ptr, count: count))
    }
    
    func setFragmentTexture(_ texture: [Float16], index: Int) {
      
        self.boundTexture = texture
    }
    
    func drawPrimitives(vertexCount: Int) {
        
    }
    
    func endEncoding() {}
}

// Entry Point
public func MGCreateSystemDefaultDevice() async -> MGDevice? {
    return await MagnesiumDevice()
}
