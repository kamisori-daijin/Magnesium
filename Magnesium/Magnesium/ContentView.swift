//
//  ContentView.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/07/11.
//
import SwiftUI
import MagnesiumKit

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
                
                // ロード中、またはデバイスがまだ準備できていない場合のオーバーレイ
                if renderContext.mgDevice == nil {
                    VStack(spacing: 16) {
                        ProgressView()
                        Text(renderContext.isLoading ? "Loading Models..." : "Initializing ANE...")
                            .foregroundColor(.secondary)
                    }
                    .frame(width: 512, height: 512)
                    .background(Color(.windowBackgroundColor).opacity(0.8))
                    .cornerRadius(12)
                }
            }
            
            if renderContext.mgDevice != nil {
                Text("3D Rasterization in progress (ANE)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .frame(width: 600, height: 680)
    }
}
