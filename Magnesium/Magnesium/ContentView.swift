//
//  ContentView.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/07/11.
//

import SwiftUI
import MagnesiumKit
import AppKit

struct ContentView: View {
    @State private var renderContext = ANERenderContext()
   
    
    var body: some View {
        VStack(spacing: 20) {
            Text("ANE 3D Renderer")
                .font(.title)
                .bold()
            
            ZStack {
                ANEMetalView(contextManager: renderContext)
                    .frame(width: 512, height: 512)
                    .cornerRadius(12)
                    .shadow(radius: 8)
                
                if renderContext.mgDevice == nil {
                    VStack(spacing: 16) {
                        if renderContext.isLoading {
                            ProgressView()
                            Text("Loading Models...")
                        } else {
                            Button("Select 3 .aimodel Files") {
                                selectFiles()
                            }
                            .buttonStyle(.glass)
                        }
                    }
                    .frame(width: 512, height: 512)
                    .cornerRadius(12)
                }
            }
        }
        .padding()
        .frame(width: 600, height: 680)
       
    }
    
    // Select File
    private func selectFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        
  
        panel.allowedFileTypes = ["aimodel"]
        
        if panel.runModal() == .OK {
            renderContext.handleSelectedURLs(panel.urls)
        }
    }
}
