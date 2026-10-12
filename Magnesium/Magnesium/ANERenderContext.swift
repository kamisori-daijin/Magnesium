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
    private var commandQueue: MTLCommandQueue?
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
           
            self.renderPipelineState = try? device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        }
    }
    
    func handleSelectedURLs(_ urls: [URL]) {
        guard let raytracerURL = urls.first(where: {
            $0.pathExtension.lowercased() == "aimodel" &&
            $0.lastPathComponent.lowercased().contains("raytracer")
        }) else {
            print("Failed to find raytracer")
            return
        }
        
        _ = raytracerURL.startAccessingSecurityScopedResource()
        
        self.isLoading = true
        Task {
            self.mgDevice = await MGCreateSystemDefaultDevice(raytracerURL: raytracerURL)
            self.isLoading = false
            
            raytracerURL.stopAccessingSecurityScopedResource()
            
            if self.mgDevice != nil {
                self.mgCommandQueue = self.mgDevice?.makeCommandQueue()
                self.update()
            }
        }
    }

    func update() {
            guard let mgDevice = self.mgDevice, !self.isComputing else { return }
            
            self.isComputing = true
            
            // Pythonのテストコードに合わせたAngleの計算
            // 60フレームで1周するアニメーションを再現
            let numFrames: Float = 60.0
            self.angle += 1.0
            if self.angle >= numFrames { self.angle = 0.0 }
            
            let currentAngle = (self.angle / numFrames) * 2.0 * Float.pi
            
            let radius: Float = 2.5
            let eyeX = radius * sin(currentAngle)
            let eyeY: Float = 1.0
            let eyeZ = radius * cos(currentAngle)
            
            mgDevice.updateCamera(
                eye: SIMD3<Float>(eyeX, eyeY, eyeZ),
                target: SIMD3<Float>(0.0, 0.0, 0.0),
                up: SIMD3<Float>(0.0, 1.0, 0.0)
            )

            // --- 動的SDFパラメータの更新 ---
            mgDevice.withSDFDataPointer { ptr in
                // 初期化
                for i in 0..<64 { ptr[i] = 0.0 }
                
                // 図形1: 中央の大きな球体
                ptr[0] = 0.0; ptr[1] = 0.0; ptr[5] = 0.4
                
                // 図形2: くり抜く用の動く直方体
                ptr[8] = 1.0; ptr[9] = 1.0
                ptr[10] = Float16(0.5 * sin(currentAngle)); ptr[13] = 0.25
                
                // 図形3: 周回するカプセル
                ptr[16] = 2.0; ptr[17] = 0.0
                ptr[18] = Float16(0.7 * cos(currentAngle)); ptr[20] = Float16(0.7 * sin(currentAngle)); ptr[21] = 0.15
                
                // 図形4: 上下するトーラス
                ptr[24] = 3.0; ptr[25] = 0.0
                ptr[27] = Float16(0.5 * sin(currentAngle * 3)); ptr[29] = 0.2
                
                // 図形5: 小さな球（合体）
                ptr[32] = 0.0; ptr[33] = 0.0
                ptr[34] = -0.4; ptr[35] = Float16(0.4 * cos(currentAngle)); ptr[37] = 0.15
                
                // 図形6: 小さな直方体（くり抜き）
                ptr[40] = 1.0; ptr[41] = 1.0
                ptr[42] = 0.4; ptr[43] = -0.4; ptr[45] = 0.15
                
                // 図形7: 斜めに動くカプセル
                ptr[48] = 2.0; ptr[49] = 0.0
                ptr[50] = Float16(0.3 * sin(currentAngle)); ptr[51] = Float16(0.3 * sin(currentAngle)); ptr[53] = 0.1
                
                // 図形8: 中心をくり抜く小さなトーラス
                ptr[56] = 3.0; ptr[57] = 1.0
                ptr[61] = 0.3
            }

            guard let mgCommandQueue = self.mgCommandQueue,
                  let mgCommandBuffer = mgCommandQueue.makeCommandBuffer() else {
                self.isComputing = false
                return
            }
            
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

            if let singleDisplayBuffer = mgDevice.getCurrentDisplayBuffer() {
                renderEncoder.setFragmentBuffer(singleDisplayBuffer, offset: 0, index: 0)
                renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }
            renderEncoder.endEncoding()
        }
        
        commandBuffer.present(drawable)
        
        commandBuffer.addCompletedHandler { _ in
            DispatchQueue.main.async { [weak self = self] in
                self?.update()
            }
        }
        
        commandBuffer.commit()
    }
}
