//
//  ANEMetalView.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/07/12.
//

import SwiftUI
import MetalKit

struct ANEMetalView: NSViewRepresentable {
    typealias NSViewType = MTKView
    
    let contextManager: ANERenderContext
    
    func makeNSView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: NSRect(x: 0, y: 0, width: 1024, height: 1024))
        
        let device: MTLDevice
        if let existingDevice = contextManager.activeDevice {
            device = existingDevice
        } else {
            guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
                fatalError("Metal device not supported.")
            }
            contextManager.setup(with: defaultDevice)
            device = defaultDevice
        }
        
        mtkView.device = device
        mtkView.delegate = context.coordinator
        mtkView.framebufferOnly = false
        
        // Synchronize completely with the context-side pipeline configuration (.bgra8Unorm)
        mtkView.colorPixelFormat = .bgra8Unorm
        
        // Ensure full support for transparent backgrounds.
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        
        mtkView.isPaused = false
        mtkView.enableSetNeedsDisplay = false
        mtkView.wantsLayer = true
        
        if let layer = mtkView.layer {
          
            layer.isOpaque = false
        }
        
        return mtkView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(manager: contextManager)
    }
    
    class Coordinator: NSObject, MTKViewDelegate {
        private let manager: ANERenderContext
        
        init(manager: ANERenderContext) {
            self.manager = manager
        }
        
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
        
        func draw(in view: MTKView) {
            manager.renderFrame(in: view)
        }
    }
}
