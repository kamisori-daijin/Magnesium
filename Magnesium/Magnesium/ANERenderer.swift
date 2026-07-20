//
//  ANERenderer.swift
//  Magnesium
//
//  Created by kamisori-daijin on 2026/07/11.
//

import Foundation
import CoreAI
import Metal
import simd

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
    private let maxVertices = 65536
    
    init(mvpURL: URL, rastURL: URL, metalDevice: MTLDevice) async throws {
        let option = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        
        self.mvpModel = try await AIModel(contentsOf: mvpURL, options: option)
        self.rastModel = try await AIModel(contentsOf: rastURL, options: option)
        
        self.mvpFunction = try mvpModel?.loadFunction(named: "main")
        self.rastFunction = try rastModel?.loadFunction(named: "main")
        
        self.vertexBufferArray = NDArray(shape: [1, 4, 1, maxVertices], scalarType: .float16)
        self.cameraMatrixArray = NDArray(shape: [4, 4], scalarType: .float16)
        
        self.outputArray = NDArray(shape: [1, 4, 1024, 1024], scalarType: .float16)
        let byteCount = 1024 * 1024 * 4 * 2
        self.displayBuffer = metalDevice.makeBuffer(length: byteCount, options: .storageModeShared)
        
        setupInitialGeometry()
    }

    private func setupInitialGeometry() {
        let vertices = geometry.getPyramidVertices()
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

    private func getEdge(pA: (Float16, Float16), pB: (Float16, Float16)) -> (Float16, Float16, Float16) {
        let A = pA.1 - pB.1
        let B = pB.0 - pA.0
        let C = -(A * pA.0 + B * pA.1)
        return (A, B, C)
    }

    private func pack(_ val: Float16) -> NDArray {
        var array = NDArray(shape: [1, 1, 1, 64], scalarType: .float16)
        var view = array.mutableView(as: Float16.self)
        view.withUnsafeMutablePointer { pointer, _, _ in
            pointer.initialize(repeating: 0, count: 64)
            pointer[0] = val
        }
        return array
    }

    private struct FaceData {
        let p0, p1, p2: (Float16, Float16)
        let invZ: Float16
    }

    func drawFrame() async throws {
        guard let mvp = mvpFunction, let rast = rastFunction else { return }
        
        // 1. MVP変換
        let mvpInputs: [String: NDArray] = ["camera_matrix": cameraMatrixArray, "vertex_buffer": vertexBufferArray]
        var mvpOutputs = try await mvp.run(inputs: mvpInputs)
        
        guard let outputValue = mvpOutputs.remove("output") else { return }
        guard var transformedArray = outputValue.ndArray else { return }
        let vertView = transformedArray.view(as: Float16.self)
        
        // 2. データの抽出（同期処理）
        var faces: [FaceData] = []
        try vertView.withUnsafePointer { vertPtr, _, _ in
            let xOffset = 0
            let yOffset = maxVertices
            let zOffset = maxVertices * 2
            
            for i in 0..<4 {
                let idx = i * 3
                let p0 = (vertPtr[xOffset + idx],     vertPtr[yOffset + idx])
                let p1 = (vertPtr[xOffset + idx + 1], vertPtr[yOffset + idx + 1])
                let p2 = (vertPtr[xOffset + idx + 2], vertPtr[yOffset + idx + 2])
                
                let zDepth = Float(vertPtr[zOffset + idx])
                let invZ = zDepth != 0 ? Float16(1.0 / zDepth) : Float16(1.0)
                
                faces.append(FaceData(p0: p0, p1: p1, p2: p2, invZ: invZ))
            }
        }
        
        // 3. ラスタライズ処理（非同期処理）
        let colors: [(Float16, Float16, Float16)] = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)]
        
        var initialView = self.outputArray.mutableView(as: Float16.self)
        initialView.withUnsafeMutablePointer { outPtr, _, _ in
            memset(outPtr, 0, 1024 * 1024 * 4 * 2)
        }
        
        for (i, face) in faces.enumerated() {
            let (A0, B0, C0) = getEdge(pA: face.p0, pB: face.p1)
            let (A1, B1, C1) = getEdge(pA: face.p1, pB: face.p2)
            let (A2, B2, C2) = getEdge(pA: face.p2, pB: face.p0)
            
            var rastInputs: [String: NDArray] = [:]
            let c = colors[i]
            
            rastInputs["a0"] = pack(A0); rastInputs["b0"] = pack(B0); rastInputs["c0"] = pack(C0)
            rastInputs["a1"] = pack(A1); rastInputs["b1"] = pack(B1); rastInputs["c1"] = pack(C1)
            rastInputs["a2"] = pack(A2); rastInputs["b2"] = pack(B2); rastInputs["c2"] = pack(C2)
            
            rastInputs["r0"] = pack(c.0); rastInputs["g0"] = pack(c.1); rastInputs["b0"] = pack(c.2)
            rastInputs["r1"] = pack(c.0); rastInputs["g1"] = pack(c.1); rastInputs["b1"] = pack(c.2)
            rastInputs["r2"] = pack(c.0); rastInputs["g2"] = pack(c.1); rastInputs["b2"] = pack(c.2)
            
            rastInputs["z0"] = pack(face.invZ); rastInputs["z1"] = pack(face.invZ); rastInputs["z2"] = pack(face.invZ)
            
            let currentOutputView = self.outputArray.mutableView(as: Float16.self)
            
            var outputViews = InferenceFunction.MutableViews()
            outputViews.insert(currentOutputView, for: "final_output")
            
            _ = try await rast.run(inputs: rastInputs, outputViews: outputViews)
        }
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
