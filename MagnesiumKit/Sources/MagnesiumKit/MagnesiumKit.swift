import Foundation
import CoreAI
import Metal

@MainActor
public protocol MGDevice: AnyObject {
    var name: String { get }
    func makeCommandQueue() -> MGCommandQueue?
    func getDisplayBuffer(index: Int) -> MTLBuffer?
    func createCameraMatrix(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> [Float16]
}

@MainActor public protocol MGCommandQueue: AnyObject { func makeCommandBuffer() -> MGCommandBuffer? }
@MainActor public protocol MGCommandBuffer: AnyObject {
    func makeRenderCommandEncoder() -> MGRenderCommandEncoder?
    func commit() async throws
}

@MainActor public protocol MGRenderCommandEncoder: AnyObject {
    func setVertexBytes(_ bytes: UnsafeRawPointer, length: Int, index: Int)
    func setFragmentTexture(_ texture: [Float16], index: Int)
    func drawPrimitives(vertexCount: Int)
    func endEncoding()
}

@MainActor
internal final class MagnesiumDevice: MGDevice {
    public let name = "MagnesiumKit"
    internal let geometry = ANE3DGeometry()
    internal var renderer: ANERenderer?
    
    public init(preURL: URL, rastURL: URL, texURL: URL) async {
        do {
            guard let systemMetalDevice = MTLCreateSystemDefaultDevice() else { return }
            self.renderer = try await ANERenderer(preURL: preURL, rastURL: rastURL, texURL: texURL, metalDevice: systemMetalDevice)
            print("Success: Renderer initialized!")
        } catch {
            print("Error: \(error)")
        }
    }
    
    public func makeCommandQueue() -> MGCommandQueue? { MagnesiumCommandQueue(device: self) }
    public func getDisplayBuffer(index: Int) -> MTLBuffer? { renderer?.displayBuffers[index] }
    public func createCameraMatrix(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> [Float16] {
        geometry.createCameraMatrix(eye: eye, target: target, up: up)
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
        guard let enc = encoder, let renderer = device.renderer else { return }
        if !enc.boundTexture.isEmpty { renderer.updateTexture(pixelData: enc.boundTexture) }
        if !enc.boundVertices.isEmpty {
            renderer.updateGeometry(vertices: enc.boundVertices, mvpWeights: enc.boundMVP, r: enc.colorsR, g: enc.colorsG, b: enc.colorsB)
        }
        try await renderer.drawFrame()
    }
}

@MainActor private final class MagnesiumRenderCommandEncoder: MGRenderCommandEncoder {
    let device: MagnesiumDevice
    var boundVertices: [Float16] = []
    var boundMVP: [Float16] = []
    var boundTexture: [Float16] = []
    var colorsR: [Float16] = []; var colorsG: [Float16] = []; var colorsB: [Float16] = []
    
    init(device: MagnesiumDevice) { self.device = device }
    
    func setVertexBytes(_ bytes: UnsafeRawPointer, length: Int, index: Int) {
            let count = length / MemoryLayout<Float16>.size
            let ptr = bytes.bindMemory(to: Float16.self, capacity: count)
            
            if index == 0 { self.boundVertices = Array(UnsafeBufferPointer(start: ptr, count: count)) }
            else if index == 1 { self.boundMVP = Array(UnsafeBufferPointer(start: ptr, count: count)) }
            else if index == 2 { self.colorsR = Array(UnsafeBufferPointer(start: ptr, count: count)) }
            else if index == 3 { self.colorsG = Array(UnsafeBufferPointer(start: ptr, count: count)) }
            else if index == 4 { self.colorsB = Array(UnsafeBufferPointer(start: ptr, count: count)) }
    }
    func setFragmentTexture(_ texture: [Float16], index: Int) { self.boundTexture = texture }
    func drawPrimitives(vertexCount: Int) {}
    func endEncoding() {}
}

@MainActor public func MGCreateSystemDefaultDevice(preURL: URL, rastURL: URL, texURL: URL) async -> MGDevice? {
    return await MagnesiumDevice(preURL: preURL, rastURL: rastURL, texURL: texURL)
}
