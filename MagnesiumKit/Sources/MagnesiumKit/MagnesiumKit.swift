import Foundation
import CoreAI
import Metal

@MainActor
public protocol MGDevice: AnyObject {
    var name: String { get }
    func makeCommandQueue() -> MGCommandQueue?
    func getDisplayBuffer() -> MTLBuffer?
    
    func updateCamera(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>, time: Float)
    
    // Texture Pointer
    func withMultiviewTexturePointer(_ body: (UnsafeMutablePointer<Float16>) -> Void)
}

@MainActor public protocol MGCommandQueue: AnyObject { func makeCommandBuffer() -> MGCommandBuffer? }

@MainActor public protocol MGCommandBuffer: AnyObject {
    func makeRenderCommandEncoder() -> MGRenderCommandEncoder?
    func commit() throws
}

@MainActor public protocol MGRenderCommandEncoder: AnyObject {
    func endEncoding()
}

@MainActor
internal final class MagnesiumDevice: MGDevice {
    public let name = "MagnesiumKit"
    internal var renderer: ANERenderer?
    
    public init(raytracerURL: URL) async {
        do {
            guard let systemMetalDevice = MTLCreateSystemDefaultDevice() else { return }
            self.renderer = try await ANERenderer(raytracerURL: raytracerURL, metalDevice: systemMetalDevice)
        } catch {
            print("Failed to Initialize: \(error)")
        }
    }
    
    public func makeCommandQueue() -> MGCommandQueue? { MagnesiumCommandQueue(device: self) }
    
    public func getDisplayBuffer() -> MTLBuffer? { renderer?.displayBuffer }
    
    // Update Animation
    public func updateCamera(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>, time: Float) {
        renderer?.updateCamera(eye: eye, target: target, up: up, time: time)
    }
    

    public func withMultiviewTexturePointer(_ body: (UnsafeMutablePointer<Float16>) -> Void) {
        guard let renderer = renderer,
              let texBuf = renderer.multiviewTextureBuffer else { return }
        
        let rawPointer = texBuf.contents().assumingMemoryBound(to: Float16.self)
        body(rawPointer)
    }
}

@MainActor private final class MagnesiumCommandQueue: MGCommandQueue {
    let device: MagnesiumDevice
    init(device: MagnesiumDevice) { self.device = device }
    func makeCommandBuffer() -> MGCommandBuffer? { MagnesiumCommandBuffer(device: device) }
}

@MainActor private final class MagnesiumCommandBuffer: MGCommandBuffer {
    let device: MagnesiumDevice
    private var encoder: MagnesiumRenderCommandEncoder?
    init(device: MagnesiumDevice) { self.device = device }
    
    func makeRenderCommandEncoder() -> MGRenderCommandEncoder? {
        let enc = MagnesiumRenderCommandEncoder(device: device)
        self.encoder = enc
        return enc
    }
    
    func commit() throws {
        guard device.renderer != nil else { return }
        
        guard let renderer = device.renderer,
              let stream = renderer.sharedComputeStream else {
            return
        }
       
        try renderer.drawFrame(onto: stream)
    }
}

@MainActor private final class MagnesiumRenderCommandEncoder: MGRenderCommandEncoder {
    let device: MagnesiumDevice
    init(device: MagnesiumDevice) { self.device = device }
    func endEncoding() {}
}

@MainActor public func MGCreateSystemDefaultDevice(raytracerURL: URL) async -> MGDevice? {
    return await MagnesiumDevice(raytracerURL: raytracerURL)
}
