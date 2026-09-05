import Foundation
import CoreAI
import Metal

@MainActor
public protocol MGDevice: AnyObject {
    var name: String { get }
    func makeCommandQueue() -> MGCommandQueue?
    func getDisplayBuffer() -> MTLBuffer?
    
    // Func
    func updateCamera(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>)
    
    // Pointer
    func withMultiviewTexturePointer(_ body: (UnsafeMutablePointer<Float16>) -> Void)
}

@MainActor public protocol MGCommandQueue: AnyObject { func makeCommandBuffer() -> MGCommandBuffer? }
@MainActor public protocol MGCommandBuffer: AnyObject {
    func makeRenderCommandEncoder() -> MGRenderCommandEncoder?
    func commit() async throws
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
    
 
    public func updateCamera(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) {
        renderer?.updateCamera(eye: eye, target: target, up: up)
    }
    

    public func withMultiviewTexturePointer(_ body: (UnsafeMutablePointer<Float16>) -> Void) {
        guard let renderer = renderer else { return }
        

        renderer.multiviewTextureArray.mutableView(as: Float16.self).withUnsafeMutablePointer { tPtr, _, _ in
            body(tPtr)
        }
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
    
    func commit() async throws {
        guard let renderer = device.renderer else { return }
        //RUn rendering
        try await renderer.drawFrame()
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
