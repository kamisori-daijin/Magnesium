//
//  ContentView.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/07/11.
//

import SwiftUI

struct ContentView: View {
    @State private var renderContext = ANERenderContext()
    
    var body: some View {
        VStack(spacing: 20) {
            Text("ANE 3D Renderer (DOOM Engine)")
                .font(.title)
                .bold()
            
            ZStack {
                // Metal view that renders continuously
                ANEMetalView(contextManager: renderContext)
                    .frame(width: 512, height: 512)
                    .cornerRadius(12)
                    .shadow(radius: 8)
                
                if renderContext.renderer == nil {
                    VStack(spacing: 16) {
                        if renderContext.isLoading {
                            ProgressView()
                            Text("Loading Models...")
                        } else {
                            Button("Select MVP & Rasterizer Models") {
                                renderContext.openModelPicker()
                            }
                            .buttonStyle(.glass)
                        }
                    }
                    .frame(width: 512, height: 512)
                    .background(Color(.windowBackgroundColor))
                    .cornerRadius(12)
                }
            }
            
            if renderContext.renderer != nil {
                VStack(spacing: 4) {
                    Text("Move: [Arrow Keys] / Fire: [Ctrl]")
                        .font(.headline)
                        .foregroundColor(.accentColor)
                    Text("Open Door: [Space] / Menu: [Enter]")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .frame(width: 600, height: 680)
        .focusable()
        .focusEffectDisabled()
        .onKeyPress(phases: [.down, .up]) { press in
            let isDown = press.phase == .down
            
            // Ignore Double Key
            func updateState(_ currentState: inout Bool) -> Bool {
                guard currentState != isDown else { return false }
                currentState = isDown
                return true
            }
            
            var handled = false
            
            switch press.key {
            case .upArrow:
                handled = updateState(&renderContext.isPressingUp)
            case .downArrow:
                handled = updateState(&renderContext.isPressingDown)
            case .leftArrow:
                handled = updateState(&renderContext.isPressingLeft)
            case .rightArrow:
                handled = updateState(&renderContext.isPressingRight)
            case .init(" "): // Space
                handled = updateState(&renderContext.isPressingSpace)
            case .return:
                handled = updateState(&renderContext.isPressingEnter)
            default:
                // Control
                if press.characters.contains(where: { $0.isASCII && $0.asciiValue == 0 }) || press.key == .init("\u{11}") {
                    handled = updateState(&renderContext.isPressingCtrl)
                } else {
                    return .ignored
                }
            }
            
            return handled ? .handled : .ignored
        }
    }
}
