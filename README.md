# Magnesium
Rasterize with ANE

high-performance 2D software rasterizer pipeline executed on the **Apple Neural Engine (ANE)** using the latest **Core AI framework (WWDC26)** and **Swift 6** 

## Features
- **Pure ANE Execution**: Edge functions and line equations are entirely executed as `f.conv2d` inside the ANE hardware pipeline.
- **Binding**: Leverages Core AI's `NDArray.View` with `withUnsafePointer` to stream continuous Planar data directly into Metal Fragment Shaders.
- **Swift 6 Concurrency & Non-Escapable (`~Escapable`) Safe**: Fully synchronized via explicit `@MainActor` task chains to prevent race conditions and uninitialized blank buffers.

## Implementation Deep Dive

### 1. The 2-Channel Static UV Input Structure
Instead of wasteful 32-channel padding, the geometry coordinates are clamped into a static `` matrix:
- **Channel 0**: X coordinates ($[-1.0 \dots 1.0]$ Grid)
- **Channel 1**: Y coordinates ($[1.0 \dots -1.0]$ Grid)

### 2. Processing Planar Data in Metal
The ANE output buffer holds raw planar data (with R, G, B, and A arranged sequentially). The Metal fragment shader samples these planes directly using precise byte offsets.
