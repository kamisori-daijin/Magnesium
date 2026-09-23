# Magnesium
ANE-powered 3D software rasterizer

A 3D graphics pipeline running on the **Apple Neural Engine (ANE)**, utilizing the latest **Core AI framework (WWDC26)**, **Metal 4**, and **Swift 6**.

<p align="center">

<video src="https://github.com/user-attachments/assets/c35d29ea-9191-423b-beba-88e668525357" width="512" height="512" autoplay loop muted playsinline></video>
</p>

## Features
- **Pure ANE-native graphics pipeline**: Geometry transformations, edge functions, line equations, and centroid texture mapping are all performed as hardware calculations within the ANE core.

- **Multi-instance parallel rendering**: Independent 3D objects with different MVP matrices can be rendered simultaneously on a single fixed ANE graph.

- **True 3D Perspective Correction Mapping**: Implemented spatial coordinate warp and pixel-level Z-depth occlusion testing on NPU hardware.

- **Metal 4 Zero Copy Britting**: Leveraged `NDArray.MutableRawView` to pipe multiplane tensor streams directly to Metal shared buffers, eliminating CPU and GPU synchronization bottlenecks.

- **AI Co-Development Infrastructure**: The majority of the Python, Swift, and Metal pipeline code was generated and rapidly prototyped using **Gemini and Siri AI**.


Implementation Details
1. Monolithic ANE pipeline
The entire graphics pipeline is structured as a single PyTorch nn.Module (ANEMonolithicPipeline), seamlessly chaining three hardware-accelerated stages:
- Texture Processing: Expands standard RGB images into a 64-channel format using a 1x1 Conv2d layer to match the batch dimension of the geometry.
- Vertex Calculation: Computes screen coordinates and edge equations for 64 independent faces simultaneously.
- Rendering: Executes perspective-correct centroid sampling and Z-depth occlusion directly on the ANE.
2. 64-batch broadcast vertex pipeline
Instead of relying on high-cost depth convolution routines, the geometry engine leverages a per-element tensor product broadcast. The input transformations are packed into a [1, 64, 4, 3] vertex tensor and a [1, 64, 4, 4] MVP weight tensor. By performing fused tensor multiplications, the ANE computes 64 intrinsic spatial transformations in parallel. The mesh grid is clamped to a static [1, 64, 256, 256] raster space.
3. Perspective corrected centroid sampling & Z-occlusion
Spatial depth inversion maps the coordinates by replacing the division denominator in the clipping space with the true spatial distance channel . The renderer calculates pixel-level depth gradients and applies a sharpness-weighted Z-buffer (z_blend_weights). It then utilizes a grouped F.conv2d operation to blend the 64 overlapping faces into a final, correctly occluded RGB output.
4. Planar zero-copy ingestion in Metal shaders
The ANE hardware dumps raw planar data (R, G, B, mask, and Z-depth arranged sequentially as separate sheets) directly into an MTLBuffer. The Metal fragment shader avoids costly memory copies by calculating precise planar offsets using the layout format stride.
```metal
// Direct plane scan within the Metal fragment shader
uint componentStride = width * height;

uint rIndex = (componentStride * 0) + pixelIndex;
uint gIndex = (componentStride * 1) + pixelIndex;
uint bIndex = (componentStride * 2) + pixelIndex;
```
---

## Known Issues
- CPU Usage: Although the render loop has been synchronized with CVDisplayLink (via MTKViewDelegate) to eliminate DispatchQueue overhead, CPU usage remains high due to per-frame color generation and related processing.
- Memory Consumption: Currently at 267MB (optimization ongoing).

---
## How to Use
1. Install Dependencies
```bash
pip install coreai-torch
```
2. Convert Shader Models
```bash
python convert_pipeline.py
```
3. Open `Magnesium.xcodeproj`
4. Build and Run
5. Use the Model Picker to select the generated `.aimodel` files.