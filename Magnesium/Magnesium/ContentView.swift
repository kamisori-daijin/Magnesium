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
            Text("ANE 3D Renderer")
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
                Text("3D Rasterization in progress (ANE)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .frame(width: 600, height: 680)
    }
}
