//
//  ANERenderer.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/07/11.
//

import Foundation
import CoreAI
import Metal
import simd // 将来的には Accelerate に移行可能

class ANERenderer {
    private var mvpModel: AIModel?
    private var rastModel: AIModel?
    
    private var mvpFunction: InferenceFunction?
    private var rastFunction: InferenceFunction?
    
    internal var vertexBufferArray: NDArray
    internal var cameraMatrixArray: NDArray
    
    private var outputArray: NDArray
    private(set) var displayBuffer: MTLBuffer?
    
    private let geometry = ANE3DGeometry()
    
    init(mvpURL: URL, rastURL: URL, metalDevice: MTLDevice) async throws {
        let option = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        
        self.mvpModel = try await AIModel(contentsOf: mvpURL, options: option)
        self.rastModel = try await AIModel(contentsOf: rastURL, options: option)
        
        self.mvpFunction = try mvpModel?.loadFunction(named: "main")
        self.rastFunction = try rastModel?.loadFunction(named: "main")
        
        self.vertexBufferArray = NDArray(shape: [1, 4, 1, 65536], scalarType: .float16)
        self.cameraMatrixArray = NDArray(shape: [4, 4], scalarType: .float16)
        
        self.outputArray = NDArray(shape: [1, 4, 1024, 1024], scalarType: .float16)
        let byteCount = 1024 * 1024 * 4 * 2
        self.displayBuffer = metalDevice.makeBuffer(length: byteCount, options: .storageModeShared)
        
        // 初期データのセットアップ
        setupInitialGeometry()
    }

    private func setupInitialGeometry() {
        let vertices = geometry.getPyramidVertices()
        
        // TODO: 将来的には Accelerate.framework (vDSP) を使用して行列計算を最適化する
        let cameraMatrix = geometry.createCameraMatrix(
            eye: SIMD3<Float>(2.0, 2.0, -5.0),
            target: SIMD3<Float>(0.0, 0.0, 0.0),
            up: SIMD3<Float>(0.0, 1.0, 0.0)
        )
        
        updateGeometry(vertices: vertices, cameraMatrix: cameraMatrix)
    }

    func updateGeometry(vertices: [Float16], cameraMatrix: [Float16]) {
        var vertexView = self.vertexBufferArray.mutableView(as: Float16.self)
        vertexView.copyElements(fromContentsOf: vertices)
        
        var cameraView = self.cameraMatrixArray.mutableView(as: Float16.self)
        cameraView.copyElements(fromContentsOf: cameraMatrix)
    }

    func drawFrame() async throws {
        guard let mvp = mvpFunction, let rast = rastFunction else { return }
        
        let mvpInputs: [String: NDArray] = [
            "camera_matrix": cameraMatrixArray,
            "vertex_buffer": vertexBufferArray
        ]
        let mvpOutputs = try await mvp.run(inputs: mvpInputs)
        
        var mutableOutputs = mvpOutputs
        guard let transformedVertices = mutableOutputs.remove("output") else { return }
        
        var rastInputs: [String: NDArray] = [:]
        // TODO: transformedVertices からエッジ(A,B,C)を計算して rastInputs にセットする処理をここに実装
        
        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(self.outputArray.mutableView(as: Float16.self), for: "final_output")
        
        _ = try await rast.run(inputs: rastInputs, outputViews: outputViews)
    }
    
    func updateDisplayBuffer(_ metalBuffer: MTLBuffer) {
        var mutableView = self.outputArray.mutableView(as: Float16.self)
        mutableView.withUnsafeMutablePointer { pointer, _, _ in
            let dest = metalBuffer.contents()
            let byteCount = 1024 * 1024 * 4 * 2
            memcpy(dest, UnsafeRawPointer(pointer), byteCount)
        }
    }
}
