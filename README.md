# Magnesium
Rasterize with ANE

high-performance 2D software rasterizer pipeline executed on the **Apple Neural Engine (ANE)** using the latest **Core AI framework (WWDC26)** and **Swift 6** 

<p align="center">
  <video src="https://github.com/user-attachments/assets/be60ad32-738f-499d-8a6a-8498b4e2c8df" width="512" height="512" autoplay loop muted playsinline></video>
</p>

## Features
- **Pure ANE Execution**: Edge functions and line equations are entirely executed as `f.conv2d` inside the ANE hardware pipeline.
- **Binding**: Leverages Core AI's `NDArray.View` with `withUnsafePointer` to stream continuous Planar data directly into Metal Fragment Shaders.
- **Swift 6 Concurrency & Non-Escapable (`~Escapable`) Safe**: Fully synchronized via explicit `@MainActor` task chains to prevent race conditions and uninitialized blank buffers.

## Implementation Deep Dive

### 1. The 2-Channel Static UV Input Structure
Instead of wasteful 32-channel padding, the geometry coordinates are clamped into a static [1, 2, 1024, 1024]
- **Channel 0**: X coordinates ($[-1.0 \dots 1.0]$ Grid)
- **Channel 1**: Y coordinates ($[1.0 \dots -1.0]$ Grid)

### 2. Processing Planar Data in Metal
The ANE output buffer holds raw planar data (with R, G, B, and A arranged sequentially). The Metal fragment shader samples these planes directly using precise byte offsets.

### How to Use
1. Install dependencies
```bash
pip install coreai-torch
```
2. Convert ShaderModel
```bash
python convert.py
```
3. Open .xcodeproj
4. Build and run
5. Select .aimodel
6. Run the app and observe the rasterized output
