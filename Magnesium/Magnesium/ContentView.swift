//
//  ContentView.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/07/11.
//

import SwiftUI
import MagnesiumKit
import AppKit // NSOpenPanelを使用するために追加

struct ContentView: View {
    @State private var renderContext = ANERenderContext()
    // isImporting の State は不要になったため削除
    
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
                                selectFiles() // 関数を呼び出す
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
        // .fileImporter は削除しました
    }
    
    // NSOpenPanelを使ってファイルを選択する関数
    private func selectFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        
        // ここで拡張子を直接指定
        panel.allowedFileTypes = ["aimodel"]
        
        if panel.runModal() == .OK {
            renderContext.handleSelectedURLs(panel.urls)
        }
    }
}
