//
//  ContentView.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/07/11.
//
import SwiftUI
public import Combine

struct ContentView: View {
    @State private var renderContext = ANERenderContext()
    
    // Start a timer on the SwiftUI side for real-time rendering (running at 60 frames per second)
    private let timer = Timer.publish(every: 1.0 / 60.0, on: .main, in: .common).autoconnect()
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Pure ANE Shader Renderer")
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
                            Text("Loading...")
                        } else {
                            Button("Select a model file.") {
                                renderContext.openModelPicker()
                            }
                            .buttonStyle(.borderedProminent)
                        }
                    }
                    .frame(width: 512, height: 512)
                    .background(Color(.windowBackgroundColor))
                    .cornerRadius(12)
                }
            }
            
          
            if renderContext.renderer != nil {
                Text("Rasterization in progress (60 FPS)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .frame(width: 600, height: 680)
      
        .onReceive(timer) { _ in
            renderContext.triggerSingleCompute()
        }
    }
}
