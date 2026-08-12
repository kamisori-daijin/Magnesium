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
    
    public init() async {
            do {
                print("MagnesiumDevice init started")
                
        
                let texURL = Bundle.main.url(forResource: "Resources/ane_texture_processor", withExtension: "aimodel")
                let rstURL = Bundle.main.url(forResource: "Resources/ane_3d_rasterizer", withExtension: "aimodel")
                let preURL = Bundle.main.url(forResource: "Resources/ane_pre_processor", withExtension: "aimodel")
                
                if let tex = texURL, let rst = rstURL, let pre = preURL {
                    guard let systemMetalDevice = MTLCreateSystemDefaultDevice() else { return }
                    
                    self.renderer = try await ANERenderer(preURL: pre, rastURL: rst, texURL: tex, metalDevice: systemMetalDevice)
                    print("Success: Renderer initialized!")
                } else {
                   
                    print("Error: Model URLs are nil. Check Package.swift resources!")
                }
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
    }
    func setFragmentTexture(_ texture: [Float16], index: Int) { self.boundTexture = texture }
    func drawPrimitives(vertexCount: Int) {}
    func endEncoding() {}
}

@MainActor public func MGCreateSystemDefaultDevice() async -> MGDevice? {
    await MagnesiumDevice()
}
