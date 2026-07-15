//
//  ANERenderer.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/07/11.
//

import Foundation
import CoreAI
import Metal

class ANERenderer {
    private var aiModel: AIModel?
    private var renderFunction: InferenceFunction?
    
    internal var uvInputArray: NDArray
    private var triWeightArray: NDArray
    private var triBiasArray: NDArray
    

    private var outputArray: NDArray
    private(set) var displayBuffer: MTLBuffer?
    
    init(modelURL: URL, metalDevice: MTLDevice) async throws {
        self.aiModel = try await AIModel(contentsOf: modelURL)
        
        guard let function = try aiModel?.loadFunction(named: "main") else {
            throw NSError(domain: "CoreAI", code: 404, userInfo: [NSLocalizedDescriptionKey: "The main function was not found."])
        }
        self.renderFunction = function
        
        // 1. Initialize UV ​​coordinates (generated from a contiguous array)
        let H = 1024, W = 1024
        let planeSize = H * W
        var uvData = [Float16](repeating: 0, count: 1 * 2 * H * W)
        
        for y in 0..<H {
            let v = Float16(1.0 - (Double(y) / 1023.0) * 2.0)
            for x in 0..<W {
                let u = Float16((Double(x) / 1023.0) * 2.0 - 1.0)
                uvData[y * W + x] = u
                uvData[planeSize + (y * W + x)] = v
            }
        }
        
        self.uvInputArray = NDArray(scalars: uvData, shape:[1,2,1024,1024])
        self.triWeightArray = NDArray(shape:[3,2,1,1], scalarType: .float16)
        self.triBiasArray = NDArray(shape:[3], scalarType: .float16)
        
       
        self.outputArray = NDArray(shape:[1,4,1024,1024], scalarType: .float16)
        let byteCount = 1024 * 1024 * 4 * 2
        self.displayBuffer = metalDevice.makeBuffer(length: byteCount, options: .storageModeShared)
       
    
    }

    func updateGeometryParameters(weights: [Float16], biases: [Float16]) {
        var weightMutableView = self.triWeightArray.mutableView(as: Float16.self)
        weightMutableView.copyElements(fromContentsOf: weights)
        
        var biasMutableView = self.triBiasArray.mutableView(as: Float16.self)
        biasMutableView.copyElements(fromContentsOf: biases)
    }

    func drawFrame() async throws {
        guard let renderFunction = self.renderFunction else { return }
        
        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(self.outputArray.mutableView(as: Float16.self), for: "relu_1")
        
     
        _ = try await renderFunction.run(inputs: ["x": uvInputArray, "tri_weight": triWeightArray, "tri_bias": triBiasArray], outputViews: outputViews)
    }
    

    func updateDisplayBuffer(_ metalBuffer: MTLBuffer) {
            var mutableView = self.outputArray.mutableView(as: Float16.self)
         
            mutableView.withUnsafeMutablePointer { pointer, _, _ in
                let dest = metalBuffer.contents()
                let byteCount = 1024 * 1024 * 4 * 2
                
                // Convert UnsafeMutablePointer to UnsafeRawPointer and pass it to memcpy
                memcpy(dest, UnsafeRawPointer(pointer), byteCount)
            }
        }
}
