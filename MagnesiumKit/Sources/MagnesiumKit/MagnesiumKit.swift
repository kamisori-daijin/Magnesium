import Foundation
import CoreAI
import Metal

@MainActor
public protocol MGDevice: AnyObject {
    var name: String { get }
    func makeCommandQueue() -> MGCommandQueue?
    func getDisplayBuffer() -> MTLBuffer?
    
    // 🌟 修正：外部からカメラ位置を渡して、デバイス内で64chアライメントハック行列を安全に直接書き込む
    func updateCamera(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>)
    
    // 🌟 修正：外部（Metal側など）に、3面図テクスチャを直接書き換えるためのポインタを安全に貸し出すブリッジ
    func withMultiviewTexturePointer(_ body: (UnsafeMutablePointer<Float16>) -> Void)
}

@MainActor public protocol MGCommandQueue: AnyObject { func makeCommandBuffer() -> MGCommandBuffer? }
@MainActor public protocol MGCommandBuffer: AnyObject {
    func makeRenderCommandEncoder() -> MGRenderCommandEncoder?
    func commit() async throws
}

// レンダラーがシンプルになったため、不要なメソッドはダミーまたは撤去可能です
@MainActor public protocol MGRenderCommandEncoder: AnyObject {
    func endEncoding()
}

@MainActor
internal final class MagnesiumDevice: MGDevice {
    public let name = "MagnesiumKit"
    internal var renderer: ANERenderer?
    
    // 🌟 修正：3つのURLから、神回路1つのraytracerURLへ集約
    public init(raytracerURL: URL) async {
        do {
            guard let systemMetalDevice = MTLCreateSystemDefaultDevice() else { return }
            self.renderer = try await ANERenderer(raytracerURL: raytracerURL, metalDevice: systemMetalDevice)
        } catch {
            print("Failed to Initialize: \(error)")
        }
    }
    
    public func makeCommandQueue() -> MGCommandQueue? { MagnesiumCommandQueue(device: self) }
    
    // 🌟 修正：バッファ配列[index]を廃止し、ゼロコピー直撃の単一バッファを返す
    public func getDisplayBuffer() -> MTLBuffer? { renderer?.displayBuffer }
    
    // 🌟 修正：安全に内部のNDArrayに対して64chアライメントハックを行う処理を呼び出す
    public func updateCamera(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) {
        renderer?.updateCamera(eye: eye, target: target, up: up)
    }
    
    // 🌟 【安全バインド移管】3面図テクスチャ（multiviewTextureArray）のポインタをここでバインドしてクロージャへ渡す
    public func withMultiviewTexturePointer(_ body: (UnsafeMutablePointer<Float16>) -> Void) {
        guard let renderer = renderer else { return }
        
        // 3次元座標と連動するようになった 1x3x256x256 のテクスチャポートのポインタを安全に展開
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
        // 8.02msのシリコン直駆動レンダリングを実行
        try await renderer.drawFrame()
    }
}

// エンコーダ側はラスタライザの仕事を失ったため、超軽量なプレースホルダーになります
@MainActor private final class MagnesiumRenderCommandEncoder: MGRenderCommandEncoder {
    let device: MagnesiumDevice
    init(device: MagnesiumDevice) { self.device = device }
    func endEncoding() {}
}

// 🌟 修正：ファクトリ関数も1つのURLで呼べるようにスッキリ化
@MainActor public func MGCreateSystemDefaultDevice(raytracerURL: URL) async -> MGDevice? {
    return await MagnesiumDevice(raytracerURL: raytracerURL)
}
