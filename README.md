# Magnesium
3D Software Rasterizer with ANE

A high-performance 3D software rasterizer pipeline executed on the **Apple Neural Engine (ANE)** using the latest **Core AI framework (WWDC26)**, **Metal 4**, and **Swift 6**.

<p align="center">
  <video src="https://github.com/user-attachments/assets/8049f5b0-200d-442f-8b67-b206751a0456" width="512" height="512" autoplay loop muted playsinline></video>
</p>

## Features
- **Pure ANE Vertex & Raster Pipeline**: MVP matrix transformations, edge functions, and line equations are entirely executed as hardware operations (e.g., `f.conv2d`) inside the ANE pipeline.
- **CPU Fallback Elimination**: Reduced CPU usage from **38% down to ~9%** by strictly bypassing CPU intervention during the main execution chain.
- **Metal 4 Tensor Binding**: Leverages Core AI's `NDArray.View` with `withUnsafePointer` to stream continuous planar tensor data directly into Metal Fragment Shaders with minimal overhead.
- **Swift 6 Concurrency & Non-Escapable (`~Escapable`) Safe**: Fully synchronized via explicit `@MainActor` task chains to prevent race conditions and uninitialized blank buffers.
- **AI-Co-Authored Infrastructure**: The majority of the Python, Swift, and Metal pipeline code was generated and fast-prototyped using **Siri AI (Apple Intelligence)**.

## Implementation Deep Dive

### 1. The 64-Batch Vertex Pipeline & 2-Channel Static UV Input
To achieve universal 3D mesh rendering without dynamic graph rebuild latency, geometry calculations support up to 65,536 vertices, processed via a **64-element batch streaming loop** on a fixed ANE graph. 

The rasterization grid is clamped into a static `[1, 2, 1024, 1024]` tensor layout:
- **Channel 0**: X coordinates ($[-1.0 \dots 1.0]$ Grid)
- **Channel 1**: Y coordinates ($[1.0 \dots -1.0]$ Grid)

### 2. Processing Planar Data in Metal
The ANE output buffer holds raw planar data (with R, G, B, and A channels arranged sequentially as separate planes). The Metal fragment shader samples these planes directly using precise byte offsets, performing zero-copy texture synthesis on the GPU.

## Known Issues / WIP
- **High Memory Footprint**: Current allocation strategy for intermediate tensor buffers and the raw planar matrix results in a massive **~5 GB memory consumption**. Optimization of the tensor lifecycle is actively under development.
- Rendering quality is still early-stage and low-resolution.

## How to Use
1. Install dependencies
```bash
pip install coreai-torch
```
2. Convert ShaderModel
```bash
python convert.py
python convert_mvp.py
```
3. Open `.xcodeproj`
4. Build and run
5. Both select the generated `.aimodel`(Click Command Key)
6. Run the app and observe the 3D rasterized output
