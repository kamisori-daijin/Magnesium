
//
//  ANERenderContext.swift
//  Magnesium
//


import Foundation
import Metal
import MetalKit
import CoreAI
internal import UniformTypeIdentifiers
import simd
// Bind

@_silgen_name("g_IsPressingW")
var g_W: Int32

@_silgen_name("g_IsPressingS")
var g_S: Int32

@_silgen_name("g_IsPressingA")
var g_A: Int32

@_silgen_name("g_IsPressingD")
var g_D: Int32

@_silgen_name("g_IsPressingLeft")
var g_Left: Int32

@_silgen_name("g_IsPressingRight")
var g_Right: Int32


@MainActor
@Observable
class ANERenderContext {
    private var angle: Float = 0.0
    private var timer: Timer?
    private(set) var renderer: ANERenderer?
    private(set) var commandQueue: MTLCommandQueue?
    private var renderPipelineState: MTLRenderPipelineState?
    
    private var sharedEvent: MTLSharedEvent?
    private var currentEventValue: UInt64 = 0
    
    var isLoading = false
    var isComputing = false
    
    //  DOOM 3D FPS State
    private var playerPosition = SIMD3<Float>(0.0, 1.0, 4.0)
    private var playerYaw: Float = Float.pi
    
    private let moveSpeed: Float = 0.15
    private let rotateSpeed: Float = 0.05
    private var doomLibHandle: UnsafeMutableRawPointer? = nil
    
    // Keyboard
    var isPressingW = false
    var isPressingS = false
    var isPressingA = false
    var isPressingD = false
    var isPressingLeft = false
    var isPressingRight = false
    
    private var doomArgs: [UnsafeMutablePointer<Int8>?] = []
    
    private let geometry = ANE3DGeometry()
    var activeDevice: MTLDevice?
    
    private var debugTextureData: [Float16] = []
    
    func setup(with device: MTLDevice) {
        self.activeDevice = device
        self.commandQueue = device.makeCommandQueue()
        self.sharedEvent = device.makeSharedEvent()
        
        self.debugTextureData = geometry.createDebugCheckerboardTexture()
        
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
    
    func openModelPicker() {
        guard let device = self.activeDevice else { return }
        let panel = NSOpenPanel()
        
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.item, .content, .data]
        panel.message = "Please select PreProcessor, Rasterizer, and Texture Processor .aimodel files."
        
        if let mainWindow = NSApplication.shared.windows.first(where: { $0.canBecomeKey }) {
            panel.beginSheetModal(for: mainWindow) { response in
                self.handlePanelResponse(response: response, panel: panel, device: device)
            }
        }
    }
    
    private func handlePanelResponse(response: NSApplication.ModalResponse, panel: NSOpenPanel, device: MTLDevice) {
        guard response == .OK, panel.urls.count == 3 else { return }
        
        let urls = panel.urls.map { $0.standardizedFileURL }
        guard let preProcessorURL = urls.first(where: { $0.lastPathComponent.lowercased().contains("pre") }),
              let rstURL = urls.first(where: { $0.lastPathComponent.lowercased().contains("rasterizer") || $0.lastPathComponent.lowercased().contains("render") }),
              let texURL = urls.first(where: { $0.lastPathComponent.lowercased().contains("texture") }) else {
            print("Error: Could not accurately identify all 3 models from filenames.")
            return
        }
        
        self.isLoading = true
        
        Task {
            defer { self.isLoading = false }
            do {
                let loadedRenderer = try await ANERenderer(preURL: preProcessorURL, rastURL: rstURL, texURL: texURL, metalDevice: device)
                self.renderer = loadedRenderer
                print("All 3 models loaded successfully.")
                
              
                let bundleWadPath = Bundle.main.path(forResource: "doom1", ofType: "wad")
                                 ?? Bundle.main.path(forResource: "DOOM1", ofType: "WAD")
                                 ?? Bundle.main.path(forResource: "DOOM1", ofType: "wad")

                if let wadPath = bundleWadPath {
                    self.doomArgs = [
                        strdup("doom"),
                        strdup("-iwad"),
                        strdup(wadPath),
                        nil
                    ]
                    
            
                    mac_Doom_Create(3, &self.doomArgs)
                    
                    print("WAD Path Linked Successfully: \(wadPath)")
                    print("DOOM Core Initialized via Source Code Direct Link.")
                } else {
                    print("Error: Faild to Load WAD Path")
                }

                
                
                self.triggerSingleCompute()
                self.startCameraRotation()
            } catch {
                print("Failed to load models: \(error)")
            }
        }
    }


    
    func triggerSingleCompute() {
        guard let renderer = self.renderer, !isComputing else { return }
        
        self.isComputing = true
        
    
        renderer.updateTexture(pixelData: self.debugTextureData)
        
        Task { @MainActor in
            do {
                try await renderer.drawFrame()
                
                self.currentEventValue += 1
                self.sharedEvent?.signaledValue = self.currentEventValue
                
            } catch {
                print("Inference error: \(error)")
            }
            
            self.isComputing = false
        }
    }

    func startCameraRotation() {
        timer?.invalidate()
     
        timer = Timer.scheduledTimer(withTimeInterval: 0.03, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self = self, let renderer = self.renderer, !self.isComputing else { return }
                
                g_W = self.isPressingW ? 1 : 0
                g_S = self.isPressingS ? 1 : 0
                g_A = self.isPressingA ? 1 : 0
                g_D = self.isPressingD ? 1 : 0
                g_Left = self.isPressingLeft ? 1 : 0
                g_Right = self.isPressingRight ? 1 : 0
                
   
                mac_Doom_Tick()
                
      
                if self.isPressingLeft {
                    self.playerYaw -= self.rotateSpeed
                }
                if self.isPressingRight {
                    self.playerYaw += self.rotateSpeed
                }
                
                let forwardX = sin(self.playerYaw)
                let forwardZ = cos(self.playerYaw)
                
                if self.isPressingW {
                    self.playerPosition.x += forwardX * self.moveSpeed
                    self.playerPosition.z += forwardZ * self.moveSpeed
                }
                if self.isPressingS {
                    self.playerPosition.x -= forwardX * self.moveSpeed
                    self.playerPosition.z -= forwardZ * self.moveSpeed
                }
                
                let eye = self.playerPosition
                let target = SIMD3<Float>(eye.x + forwardX, eye.y, eye.z + forwardZ)
                
                let cameraMatrix = self.geometry.createCameraMatrix(
                    eye: eye,
                    target: target,
                    up: SIMD3<Float>(0.0, 1.0, 0.0)
                )

                var mvpWeights = [Float16](repeating: 0.0, count: 4 * 4 * 64)
                var vertices = [Float16](repeating: 0.0, count: 1 * 4 * 3 * 64)
                var colorsR = [Float16](repeating: 0.0, count: 64)
                var colorsG = [Float16](repeating: 0.0, count: 64)
                var colorsB = [Float16](repeating: 0.0, count: 64)
                
                let wChannelOffset = 3 * 3 * 64
                for faceIdx in 0..<64 {
                    vertices[wChannelOffset + (0 * 64) + faceIdx] = 1.0
                    vertices[wChannelOffset + (1 * 64) + faceIdx] = 1.0
                    vertices[wChannelOffset + (2 * 64) + faceIdx] = 1.0
                }

                // ① 前後の壁データ（Z = -4.0 の奥の壁 ＆ Z = 4.0 の手前の壁）
                let northSouthWalls: [[[Float16]]] = [
                    [[ -3.0,  2.0, -4.0, 1.0], [ -3.0,  0.0, -4.0, 1.0], [  3.0,  0.0, -4.0, 1.0]],
                    [[ -3.0,  2.0, -4.0, 1.0], [  3.0,  0.0, -4.0, 1.0], [  3.0,  2.0, -4.0, 1.0]],
                    [[  3.0,  2.0,  4.0, 1.0], [  3.0,  0.0,  4.0, 1.0], [ -3.0,  0.0,  4.0, 1.0]],
                    [[  3.0,  2.0,  4.0, 1.0], [ -3.0,  0.0,  4.0, 1.0], [ -3.0,  2.0,  4.0, 1.0]],
                ]
                
                // ② 左右の壁データ（X = -3.0 の左の壁 ＆ X = 3.0 の右の壁）
                let eastWestWalls: [[[Float16]]] = [
                    [[ -3.0,  2.0,  4.0, 1.0], [ -3.0,  0.0,  4.0, 1.0], [ -3.0,  0.0, -4.0, 1.0]],
                    [[ -3.0,  2.0,  4.0, 1.0], [ -3.0,  0.0, -4.0, 1.0], [ -3.0,  2.0, -4.0, 1.0]],
                    [[  3.0,  2.0, -4.0, 1.0], [  3.0,  0.0, -4.0, 1.0], [  3.0,  0.0,  4.0, 1.0]],
                    [[  3.0,  2.0, -4.0, 1.0], [  3.0,  0.0,  4.0, 1.0], [  3.0,  2.0,  4.0, 1.0]],
                ]
                
                let wallColors: [(Float16, Float16, Float16)] = [
                    (0.5, 0.4, 0.4), (0.5, 0.4, 0.4), (0.4, 0.4, 0.5), (0.4, 0.4, 0.5)
                ]

                // スロット0〜3番に「前後の壁」を建築
                for i in 0..<4 {
                    let slot = i
                    colorsR[slot] = wallColors[i].0; colorsG[slot] = wallColors[i].1; colorsB[slot] = wallColors[i].2
                    
                    for v in 0..<3 {
                        for ch in 0..<4 {
                            let pIndex = (ch * 3 * 64) + (v * 64) + slot
                            vertices[pIndex] = northSouthWalls[i][v][ch]
                        }
                    }
                    for m in 0..<16 {
                        mvpWeights[m * 64 + slot] = cameraMatrix[m]
                    }
                }

                // スロット4〜7番に「左右の壁」を建築
                for i in 0..<4 {
                    let slot = 4 + i
                    colorsR[slot] = wallColors[i].0; colorsG[slot] = wallColors[i].1; colorsB[slot] = wallColors[i].2
                    
                    for v in 0..<3 {
                        for ch in 0..<4 {
                            let pIndex = (ch * 3 * 64) + (v * 64) + slot
                            vertices[pIndex] = eastWestWalls[i][v][ch]
                        }
                    }
                    for m in 0..<16 {
                        mvpWeights[m * 64 + slot] = cameraMatrix[m]
                    }
                }

                // 💡 STEP 5: すべてが揃った状態で ANE へのパイプライン転送を点火！
                renderer.updateGeometry(
                    vertices: vertices,
                    mvpWeights: mvpWeights,
                    r: colorsR,
                    g: colorsG,
                    b: colorsB
                )
                self.triggerSingleCompute()
            }
        }
    }
  
    func renderFrame(in view: MTKView) {
        view.colorPixelFormat = .bgra8Unorm
        
        guard let renderer = self.renderer,
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
           
    
            if let buffer = renderer.displayBuffers[0] {
                renderEncoder.setFragmentBuffer(buffer, offset: 0, index: 0)
                renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }
            renderEncoder.endEncoding()
        }
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}

