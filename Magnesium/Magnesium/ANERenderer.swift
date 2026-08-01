//
//  ANERenderer.swift
//  Magnesium
//

import Foundation
import CoreAI
import Metal
import simd

class ANERenderer {
    private var mvpModel: AIModel?
    private var rastModel: AIModel?
    private var texModel: AIModel?
    
    private var mvpFunction: InferenceFunction?
    private var rastFunction: InferenceFunction?
    private var texFunction: InferenceFunction?
    
    internal var vertexBufferArray: NDArray
    internal var cameraMatrixArray: NDArray
    
    internal var rawTextureArray: NDArray
    internal var alignedTextureArray: NDArray
    internal var uvBufferArray: NDArray
    
    private var metalHeap: MTLHeap?
    private(set) var displayBuffers: [MTLBuffer?] = [nil, nil, nil, nil]
    
    private let geometry = ANE3DGeometry()
    private let maxVertices = 65536
    private let metalDevice: MTLDevice
    
    private let layerByteCount = 64 * 1 * 256 * 256 * 2
    
    private var packedParamCache: [String: NDArray] = [:]
    
    init(mvpURL: URL, rastURL: URL, texURL: URL, metalDevice: MTLDevice) async throws {
        self.metalDevice = metalDevice
        let option = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        
        self.mvpModel = try await AIModel(contentsOf: mvpURL, options: option)
        self.rastModel = try await AIModel(contentsOf: rastURL, options: option)
        self.texModel = try await AIModel(contentsOf: texURL, options: option)
        
        self.mvpFunction = try mvpModel?.loadFunction(named: "main")
        self.rastFunction = try rastModel?.loadFunction(named: "main")
        self.texFunction = try texModel?.loadFunction(named: "main")
        
        self.vertexBufferArray = NDArray(shape: [1, 4, 1, maxVertices], scalarType: .float16)
        self.cameraMatrixArray = NDArray(shape: [4, 4], scalarType: .float16)
        
        self.rawTextureArray = NDArray(shape:[1,3,256,256], scalarType: .float16)
        self.alignedTextureArray = NDArray(shape:[1,64,256,256], scalarType: .float16)
        self.uvBufferArray = NDArray(shape: [1, 2, 1, maxVertices], scalarType: .float16)
        
        let paramKeys = ["a0", "b0", "c0", "a1", "b1", "c1", "a2", "b2", "c2",
                         "r0", "g0", "b0_col", "r1", "g1", "b1_col", "r2", "g2", "b2_col", "z_weight"]
        for key in paramKeys {
            self.packedParamCache[key] = NDArray(shape: [1,1,1,64], scalarType: .float16)
        }
        
        setupMetalHeap()
        setupInitialGeometry()
    }

    private func setupMetalHeap() {
        let singleDisplayBufferSize = layerByteCount * 4
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

    private func setupInitialGeometry() {
        let vertices = geometry.getPyramidVertices()
        let cameraMatrix = geometry.createCameraMatrix(
            eye: SIMD3<Float>(2.0, 2.0, -5.0),
            target: SIMD3<Float>(0.0, 0.0, 0.0),
            up: SIMD3<Float>(0.0, 1.0, 0.0)
        )
        let uvs = geometry.getPyramidUVs()
        updateGeometry(vertices: vertices, cameraMatrix: cameraMatrix, uvs: uvs)
    }

    func updateGeometry(vertices: [Float16], cameraMatrix: [Float16], uvs: [Float16]) {
        var vertexView = self.vertexBufferArray.mutableView(as: Float16.self)
        vertexView.copyElements(fromContentsOf: vertices)
        
        var cameraView = self.cameraMatrixArray.mutableView(as: Float16.self)
        cameraView.copyElements(fromContentsOf: cameraMatrix)
        
        var uvView = self.uvBufferArray.mutableView(as: Float16.self)
        uvView.copyElements(fromContentsOf: uvs)
    }
    
    func updateTexture(pixelData: [Float16]) {
        var texView = self.rawTextureArray.mutableView(as: Float16.self)
        texView.copyElements(fromContentsOf: pixelData)
    }

    private func getEdge(pA: (Float16, Float16), pB: (Float16, Float16)) -> (Float16, Float16, Float16) {
        let A = pA.1 - pB.1
        let B = pB.0 - pA.0
        let C = -(A * pA.0 + B * pA.1)
        return (A, B, C)
    }

    private func pack(_ val: Float16, into key: String) -> NDArray {
        guard var array = packedParamCache[key] else {
            return NDArray(shape:[1,1,1,64], scalarType: .float16)
        }
        
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
        guard let mvp = mvpFunction, let rst = rastFunction, let tex = texFunction else { return }
        
        let texInputs: [String: NDArray] = ["raw_image": rawTextureArray]
        
        var texOutputViews = InferenceFunction.MutableViews()
        let texDestView = alignedTextureArray.mutableView(as: Float16.self)
        texOutputViews.insert(texDestView, for: "convolution")
        
        let _ = try await tex.run(inputs: texInputs, outputViews: texOutputViews)
        
        let mvpInputs: [String: NDArray] = ["camera_matrix": cameraMatrixArray, "vertex_buffer": vertexBufferArray]
        var mvpOutputs = try await mvp.run(inputs: mvpInputs)
        
        guard let outputValue = mvpOutputs.remove("cat") else { return }
        guard var transformedArray = outputValue.ndArray else { return }
        let vertView = transformedArray.view(as: Float16.self)
        
        var faces: [FaceData] = []
        try vertView.withUnsafePointer { vertPtr, _, _ in
            for i in 0..<4 {
                let idx = i * 3
                let p0 = (vertPtr[0 * maxVertices + idx],     vertPtr[1 * maxVertices + idx])
                let p1 = (vertPtr[0 * maxVertices + idx + 1], vertPtr[1 * maxVertices + idx + 1])
                let p2 = (vertPtr[0 * maxVertices + idx + 2], vertPtr[1 * maxVertices + idx + 2])
                
                let zDepth = Float(vertPtr[2 * maxVertices + idx])
                let invZ = zDepth != 0 ? Float16(1.0 / zDepth) : Float16(1.0)
                faces.append(FaceData(p0: p0, p1: p1, p2: p2, invZ: invZ))
            }
        }
        
        let colors: [(Float16, Float16, Float16)] = [
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)
        ]
        
        for (i, face) in faces.enumerated() {
            let (A0, B0, C0) = getEdge(pA: face.p0, pB: face.p1)
            let (A1, B1, C1) = getEdge(pA: face.p1, pB: face.p2)
            let (A2, B2, C2) = getEdge(pA: face.p2, pB: face.p0)
            
            var rstInputs: [String: NDArray] = [:]
            let c = colors[i]
            
            rstInputs["a0"] = pack(A0, into: "a0"); rstInputs["b0"] = pack(B0, into: "b0"); rstInputs["c0"] = pack(C0, into: "c0")
            rstInputs["a1"] = pack(A1, into: "a1"); rstInputs["b1"] = pack(B1, into: "b1"); rstInputs["c1"] = pack(C1, into: "c1")
            rstInputs["a2"] = pack(A2, into: "a2"); rstInputs["b2"] = pack(B2, into: "b2"); rstInputs["c2"] = pack(C2, into: "c2")
            
            rstInputs["r0"] = pack(c.0, into: "r0"); rstInputs["g0"] = pack(c.1, into: "g0"); rstInputs["b0_col"] = pack(c.2, into: "b0_col")
            rstInputs["r1"] = pack(c.0, into: "r1"); rstInputs["g1"] = pack(c.1, into: "g1"); rstInputs["b1_col"] = pack(c.2, into: "b1_col")
            rstInputs["r2"] = pack(c.0, into: "r2"); rstInputs["g2"] = pack(c.1, into: "g2"); rstInputs["b2_col"] = pack(c.2, into: "b2_col")
            
            rstInputs["z_weight"] = pack(face.invZ, into: "z_weight")
            rstInputs["processed_texture"] = self.alignedTextureArray
            
            guard let metalBuf = self.displayBuffers[i] else { continue }
            
            var outputViews = InferenceFunction.MutableViews()
            
            let viewForR = NDArray.MutableRawView(metalBuffer: metalBuf, byteOffset: layerByteCount * 0, scalarType: .float16, shape: [64, 1, 256, 256]).view(as: Float16.self)
            outputViews.insert(viewForR, for: "convolution_3")
            
            let viewForG = NDArray.MutableRawView(metalBuffer: metalBuf, byteOffset: layerByteCount * 1, scalarType: .float16, shape: [64, 1, 256, 256]).view(as: Float16.self)
            outputViews.insert(viewForG, for: "convolution_4")
            
            let viewForB = NDArray.MutableRawView(metalBuffer: metalBuf, byteOffset: layerByteCount * 2, scalarType: .float16, shape: [64, 1, 256, 256]).view(as: Float16.self)
            outputViews.insert(viewForB, for: "convolution_5")
            
            let viewForMask = NDArray.MutableRawView(metalBuffer: metalBuf, byteOffset: layerByteCount * 3, scalarType: .float16, shape: [64, 1, 256, 256]).view(as: Float16.self)
            outputViews.insert(viewForMask, for: "convolution_6")

            let _ = try await rst.run(inputs: rstInputs, outputViews: outputViews)
        }
    }
}
