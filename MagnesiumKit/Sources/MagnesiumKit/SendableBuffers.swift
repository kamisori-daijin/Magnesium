//
//  SendableBuffers.swift
//  MagnesiumKit
//
//  Created by kamisori-daijin on 2026/08/21.
//
import Metal

struct SendableBuffers: @unchecked Sendable {
    var buffers: [MTLBuffer?] = [nil, nil, nil, nil]
    
    subscript(index: Int) -> MTLBuffer? {
        get { buffers[index] }
        set { buffers[index] = newValue }
    }
}
