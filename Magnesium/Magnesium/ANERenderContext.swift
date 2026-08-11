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
@_silgen_name("g_IsPressingUp") var g_Up: Int32
@_silgen_name("g_IsPressingDown") var g_Down: Int32
@_silgen_name("g_IsPressingLeft") var g_Left: Int32
@_silgen_name("g_IsPressingRight") var g_Right: Int32
@_silgen_name("g_IsPressingCtrl") var g_Ctrl: Int32
@_silgen_name("g_IsPressingSpace") var g_Space: Int32
@_silgen_name("g_IsPressingEnter") var g_Enter: Int32

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
    
    private var doomLibHandle: UnsafeMutableRawPointer? = nil
    
    // Keyboard
    var isPressingUp = false
    var isPressingDown = false
    var isPressingLeft = false
    var isPressingRight = false
    var isPressingCtrl = false
    var isPressingSpace = false
    var isPressingEnter = false
    
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
              let rstURL = urls.first(where: { $0.lastPathComponent.lowercased().contains("rasterizer") }),
              let texURL = urls.first(where: { $0.lastPathComponent.lowercased().contains("texture") }) else { return }
        
        self.isLoading = true
        Task {
            defer { self.isLoading = false }
            do {
                self.renderer = try await ANERenderer(preURL: preProcessorURL, rastURL: rstURL, texURL: texURL, metalDevice: device)
                if let wadPath = Bundle.main.path(forResource: "DOOM1", ofType: "WAD") {
                    self.doomArgs = [ strdup("doom"), strdup("-iwad"), strdup(wadPath), nil ]
                    mac_Doom_Create(3, &self.doomArgs)
                }
                self.triggerSingleCompute()
                self.startCameraRotation()
            } catch {}
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
            } catch {}
            self.isComputing = false
        }
    }

    func startCameraRotation() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.03, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self = self, let renderer = self.renderer, !self.isComputing else { return }
                
                g_Up = self.isPressingUp ? 1 : 0
                g_Down = self.isPressingDown ? 1 : 0
                g_Left = self.isPressingLeft ? 1 : 0
                g_Right = self.isPressingRight ? 1 : 0
                g_Ctrl = self.isPressingCtrl ? 1 : 0
                g_Space = self.isPressingSpace ? 1 : 0
                g_Enter = self.isPressingEnter ? 1 : 0
                
                mac_Doom_Tick()
                
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

                let northSouthWalls: [[[Float16]]] = [
                    [[ -3.2, -2.0, -4.0, 1.0], [  3.2, -2.0, -4.0, 1.0], [ -3.2,  2.0, -4.0, 1.0]],
                    [[  3.2, -2.0, -4.0, 1.0], [  3.2,  2.0, -4.0, 1.0], [ -3.2,  2.0, -4.0, 1.0]],
                ]

                for i in 0..<2 {
                    colorsR[i] = 1.0; colorsG[i] = 1.0; colorsB[i] = 1.0
                    for v in 0..<3 {
                        for ch in 0..<4 { vertices[(ch * 3 * 64) + (v * 64) + i] = northSouthWalls[i][v][ch] }
                    }
                    mvpWeights[0 * 64 + i] = 1.0
                    mvpWeights[5 * 64 + i] = 1.0
                    mvpWeights[10 * 64 + i] = 1.0
                    mvpWeights[15 * 64 + i] = 1.0
                }

                renderer.updateTexture(pixelData: self.debugTextureData)
                renderer.updateGeometry(vertices: vertices, mvpWeights: mvpWeights, r: colorsR, g: colorsG, b: colorsB)
                self.triggerSingleCompute()
            }
        }
    }
  
    func renderFrame(in view: MTKView) {
        view.colorPixelFormat = .bgra8Unorm
        guard let renderer = self.renderer, let queue = self.commandQueue, let pipeline = self.renderPipelineState,
              let sharedEvent = self.sharedEvent, let renderPassDescriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable, let commandBuffer = queue.makeCommandBuffer() else { return }
        
        if self.currentEventValue > 0 { commandBuffer.encodeWaitForEvent(sharedEvent, value: self.currentEventValue) }

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
