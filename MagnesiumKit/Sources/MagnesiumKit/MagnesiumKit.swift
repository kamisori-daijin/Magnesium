import Foundation
import CoreAI
import Metal

@MainActor
public protocol MGDevice: AnyObject {
    var name: String { get }
    func makeCommandQueue() -> MGCommandQueue?
    func getDisplayBuffer(index: Int) -> MTLBuffer?
    func createCameraMatrix(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> [Float16]
    
    // 【追加】ゼロコピー用のポインタアクセス
    func withGeometryPointers(_ body: (UnsafeMutablePointer<Float16>, UnsafeMutablePointer<Float16>, UnsafeMutablePointer<Float16>, UnsafeMutablePointer<Float16>, UnsafeMutablePointer<Float16>) -> Void)
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
    
    // 【追加】ANERendererのポインタを直接公開
    public func withGeometryPointers(_ body: (UnsafeMutablePointer<Float16>, UnsafeMutablePointer<Float16>, UnsafeMutablePointer<Float16>, UnsafeMutablePointer<Float16>, UnsafeMutablePointer<Float16>) -> Void) {
        guard let renderer = renderer else { return }
        
        renderer.expandedVerticesArray.mutableView(as: Float16.self).withUnsafeMutablePointer { vPtr, _, _ in
            renderer.mvpWeightsArray.mutableView(as: Float16.self).withUnsafeMutablePointer { mPtr, _, _ in
                renderer.colorsRArray.mutableView(as: Float16.self).withUnsafeMutablePointer { rPtr, _, _ in
                    renderer.colorsGArray.mutableView(as: Float16.self).withUnsafeMutablePointer { gPtr, _, _ in
                        renderer.colorsBArray.mutableView(as: Float16.self).withUnsafeMutablePointer { bPtr, _, _ in
                            body(vPtr, mPtr, rPtr, gPtr, bPtr)
                        }
                    }
                }
            }
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
        // 【修正】データは既にポインタ経由で書き込まれているため、描画のみを実行
        try await renderer.drawFrame()
    }
}

@MainActor private final class MagnesiumRenderCommandEncoder: MGRenderCommandEncoder {
    let device: MagnesiumDevice
    
    init(device: MagnesiumDevice) { self.device = device }
    
    // 【修正】ゼロコピー化により、ここでのデータ保持は不要に
    func setVertexBytes(_ bytes: UnsafeRawPointer, length: Int, index: Int) {}
    func setFragmentTexture(_ texture: [Float16], index: Int) {}
    func drawPrimitives(vertexCount: Int) {}
    func endEncoding() {}
}

@MainActor public func MGCreateSystemDefaultDevice(preURL: URL, rastURL: URL, texURL: URL) async -> MGDevice? {
    return await MagnesiumDevice(preURL: preURL, rastURL: rastURL, texURL: texURL)
}
