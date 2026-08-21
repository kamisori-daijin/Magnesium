//
//  ANERenderer+Setup.swift
//  MagnesiumKit
//
//  Created by kamisori-daijin on 2026/08/21.
//

import Foundation
import CoreAI
import Metal


@MainActor extension ANERenderer {
    
    internal func setupMetalHeap() {
        let singlePlaneSize = tileSizeInBytes * 192
        let singleDisplayBufferSize = singlePlaneSize * 4
        let totalRequiredMemory = singleDisplayBufferSize * 4
        
        let heapDescriptor = MTLHeapDescriptor()
        heapDescriptor.size = totalRequiredMemory
        heapDescriptor.storageMode = .shared
        heapDescriptor.type = .placement
        
        self.metalHeap = metalDevice.makeHeap(descriptor: heapDescriptor)
        
        guard let heap = self.metalHeap else { return }
        for i in 0..<4 {
            self.displayBuffers[i] = heap.makeBuffer(
                length: singleDisplayBufferSize,
                options: .storageModeShared,
                offset: i * singleDisplayBufferSize
            )
        }
    }
    
    internal func setupInitialGeometry() {
        let vertices = geometry.getDummyVertices()
        let mvp = geometry.getDummyMVPWeights()
        let color = geometry.getDummyColor()
        updateGeometry(vertices: vertices, mvpWeights: mvp, r: color, g: color, b: color)
        
        let debugTexture = geometry.createDebugCheckerboardTexture()
        updateTexture(pixelData: debugTexture)
    }
    
    func updateGeometry(vertices: [Float16], mvpWeights: [Float16], r: [Float16], g: [Float16], b: [Float16]) {
        var vertexView = self.expandedVerticesArray.mutableView(as: Float16.self)
        vertexView.copyElements(fromContentsOf: vertices)
        
        var mvpView = self.mvpWeightsArray.mutableView(as: Float16.self)
        mvpView.copyElements(fromContentsOf: mvpWeights)
        
        var rView = self.colorsRArray.mutableView(as: Float16.self)
        rView.copyElements(fromContentsOf: r)
        
        var gView = self.colorsGArray.mutableView(as: Float16.self)
        gView.copyElements(fromContentsOf: g)
        
        var bView = self.colorsBArray.mutableView(as: Float16.self)
        bView.copyElements(fromContentsOf: b)
    }
    
    func updateTexture(pixelData: [Float16]) {
        var texView = self.rawTextureArray.mutableView(as: Float16.self)
        let expectedCount = 1 * 3 * 128 * 128
        
        if pixelData.count != expectedCount {
            let debugData = geometry.createDebugCheckerboardTexture()
            texView.copyElements(fromContentsOf: debugData)
        } else {
            texView.copyElements(fromContentsOf: pixelData)
        }
    }
}
