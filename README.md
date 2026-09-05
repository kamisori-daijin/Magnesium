# Magnesium
ANE-powered 3D software rasterizer

A 3D graphics pipeline running on the **Apple Neural Engine (ANE)**, utilizing the latest **Core AI framework (WWDC26)**, **Metal 4**, and **Swift 6**.

<p align="center">

<video src="https://github.com/user-attachments/assets/de394641-edd3-47c0-baef-c1a5f306357d" width="512" height="512" autoplay loop muted playsinline></video>
</p>

## Features
- **Pure ANE-native graphics pipeline**: Geometry transformations, edge functions, line equations, and centroid texture mapping are all performed as hardware calculations within the ANE core.

- **Multi-instance parallel rendering**: Independent 3D objects with different MVP matrices can be rendered simultaneously on a single fixed ANE graph.

- **True 3D Perspective Correction Mapping**: Implemented spatial coordinate warp and pixel-level Z-depth occlusion testing on NPU hardware.

- **Metal 4 Zero Copy Britting**: Leveraged `NDArray.MutableRawView` to pipe multiplane tensor streams directly to Metal shared buffers, eliminating CPU and GPU synchronization bottlenecks.

- **AI Co-Development Infrastructure**: The majority of the Python, Swift, and Metal pipeline code was generated and rapidly prototyped using **Gemini and Siri AI**.


## Implementation Details

### 1. 64-Batch Broadcast Vertex Pipeline
To render multiple independent objects without triggering the high-cost depth convolution (`groups=64`) routines that would confuse the Core AI compiler, the geometry engine leverages a **per-element tensor product broadcast hack**.

The input transformations are packed into a `[1, 4, 4, 1, 64]` tensor representing 64 independent 4x4 MVP matrices. By performing a fused `torch.sum(*)` operation, the ANE multifires the 64 intrinsic spatial transformations in parallel. The mesh grid is clamped to a static `[1, 2, 256, 256]` raster space.
- **Channel 0**: X coordinate ($[-1.0 \dots 1.0]$ grid)
- **Channel 1**: Y coordinate ($[1.0 \dots -1.0]$ grid)

### 2. Perspective Corrected Centroid Sampling
Spatial depth inversion maps the coordinates by replacing the division denominator in the clipping space with the true spatial distance channel $W_c$. The 3D geometry engine outputs precise 3-vertex inverse depth gradients via hidden tensor blocks (`slice_11` to `slice_13`), and the rasterizer constructs smooth, pixel-level depth gradients to achieve overlap occlusion.

### 3. Planar Zero-Copy Ingestion in Metal Shaders
The ANE hardware dumps raw planar data (R, G, B, and mask arranged sequentially as separate sheets) directly into an `MTLBuffer` allocated on the heap. The Metal fragment shader avoids costly memory copies and achieves high-overhead texturing by calculating precise planar offsets using the layout format stride.

```metal
// Direct plane scan within the Metal fragment shader
uint componentStride = 64 * width * height;

uint rIndex = (componentStride * 0) + pixelIndex;

uint gIndex = (componentStride * 1) + pixelIndex;

uint bIndex = (componentStride * 2) + pixelIndex;

```

---

## Known Issues
Memory consumption is still high at 267MB, and CPU usage is around 38%.

---
## How to Use
1. Install Dependencies
```bash
pip install coreai-torch
```
2. Convert Shader Models
```bash
python convert.py
python convert_prepro.py
python convert_texture.py
```
3. Open `Magnesium.xcodeproj`
4. Build and Run
5. Use the Model Picker to select the three generated `.aimodel` files (to select multiple assets, hold down the Command key while selecting).