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
                            // Liquid Glass Button
                            .buttonStyle(.glass)
                        }
                    }
                    .frame(width: 512, height: 512)
                    .background(Color(.windowBackgroundColor))
                    .cornerRadius(12)
                }
            }
            
            if renderContext.renderer != nil {
               
                Text(" Move: [W][S] / Turn: [A][D] or [Arrow Keys]")
                    .font(.headline)
                    .foregroundColor(.accentColor)
            }
        }
        .padding()
        .frame(width: 600, height: 680)
        

        .focusable()
        .focusEffectDisabled()
        .onKeyPress(phases: [.down, .up]) { press in
            let isDown = press.phase == .down
            
            // 押し下げ（.down）なら true、離されたら（.up）なら false をフラグに叩き込む
            switch press.key {
            case .init("w"), .init("W"):
                renderContext.isPressingW = isDown
            case .init("s"), .init("S"):
                renderContext.isPressingS = isDown
            case .init("a"), .init("A"):
                renderContext.isPressingA = isDown
            case .init("d"), .init("D"):
                renderContext.isPressingD = isDown
            case .leftArrow:
                renderContext.isPressingLeft = isDown
            case .rightArrow:
                renderContext.isPressingRight = isDown
            case .return:
                renderContext.isPressingEnter = isDown

            case .space:
                renderContext.isPressingSpace = isDown

            default:
                return .ignored
            }
            return .handled
        }
    }
}
