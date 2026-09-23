import Foundation
import CoreAI
import Metal

@MainActor
public protocol MGDevice: AnyObject {
    var name: String { get }
    func makeCommandQueue() -> MGCommandQueue?
    func getDisplayBuffer(index: Int) -> MTLBuffer?
    func createCameraMatrix(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> [Float16]
    
    func withGeometryPointers(_ body: (UnsafeMutablePointer<Float16>, UnsafeMutablePointer<Float16>, UnsafeMutablePointer<Float16>, UnsafeMutablePointer<Float16>, UnsafeMutablePointer<Float16>) -> Void)
}

@MainActor public protocol MGCommandQueue: AnyObject { func makeCommandBuffer() -> MGCommandBuffer? }
@MainActor public protocol MGCommandBuffer: AnyObject {
    func makeRenderCommandEncoder() -> MGRenderCommandEncoder?
    func commit() async throws
}

@MainActor public protocol MGRenderCommandEncoder: AnyObject {
    func setVertexBytes(_ bytes: UnsafeRawPointer, length: Int, index: Int)
    func withFragmentTexturePointer(index: Int, _ body: (UnsafeMutablePointer<Float16>) -> Void)
    func drawPrimitives(vertexCount: Int)
    func endEncoding()
}

@MainActor
internal final class MagnesiumDevice: MGDevice {
    public let name = "MagnesiumKit"
    internal let geometry = MGUtil()
    internal var renderer: ANERenderer?
    
    public init(modelURL: URL) async {
        do {
            guard let systemMetalDevice = MTLCreateSystemDefaultDevice() else { return }
            self.renderer = try await ANERenderer(modelURL: modelURL, metalDevice: systemMetalDevice)
        } catch {
            print("Error initializing ANERenderer: \(error)")
        }
    }
    
    public func makeCommandQueue() -> MGCommandQueue? { MagnesiumCommandQueue(device: self) }
    public func getDisplayBuffer(index: Int) -> MTLBuffer? { renderer?.displayBuffers[index] }
    public func createCameraMatrix(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> [Float16] {
        geometry.createCameraMatrix(eye: eye, target: target, up: up)
    }
    
    public func withGeometryPointers(_ body: (UnsafeMutablePointer<Float16>, UnsafeMutablePointer<Float16>, UnsafeMutablePointer<Float16>, UnsafeMutablePointer<Float16>, UnsafeMutablePointer<Float16>) -> Void) {
        guard let renderer = renderer,
              let vBuf = renderer.expandedVerticesBuffer,
              let mBuf = renderer.mvpWeightsBuffer,
              let rBuf = renderer.colorsRBuffer,
              let gBuf = renderer.colorsGBuffer,
              let bBuf = renderer.colorsBBuffer else { return }
        
        let vPtr = vBuf.contents().assumingMemoryBound(to: Float16.self)
        let mPtr = mBuf.contents().assumingMemoryBound(to: Float16.self)
        let rPtr = rBuf.contents().assumingMemoryBound(to: Float16.self)
        let gPtr = gBuf.contents().assumingMemoryBound(to: Float16.self)
        let bPtr = bBuf.contents().assumingMemoryBound(to: Float16.self)
        
        body(vPtr, mPtr, rPtr, gPtr, bPtr)
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
        // ComputeStream を渡してエンコード
        try renderer.drawFrame(onto: renderer.sharedComputeStream)
    }
}

@MainActor private final class MagnesiumRenderCommandEncoder: MGRenderCommandEncoder {
    let device: MagnesiumDevice
    
    init(device: MagnesiumDevice) { self.device = device }
    
    func setVertexBytes(_ bytes: UnsafeRawPointer, length: Int, index: Int) {}
    
    func withFragmentTexturePointer(index: Int, _ body: (UnsafeMutablePointer<Float16>) -> Void) {
        guard let renderer = device.renderer,
              let tBuf = renderer.rawTextureBuffer else { return }
        
        let tPtr = tBuf.contents().assumingMemoryBound(to: Float16.self)
        body(tPtr)
    }
    
    func drawPrimitives(vertexCount: Int) {}
    func endEncoding() {}
}

@MainActor public func MGCreateSystemDefaultDevice(modelURL: URL) async -> MGDevice? {
    return await MagnesiumDevice(modelURL: modelURL)
}
