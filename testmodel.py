import os
import torch
import numpy as np
import torchvision.utils as vutils
from RayTracingCore import ANERayTracingCore

def create_inverse_view_matrix(eye, target, up):
    """
    Inverse view matrix for ray tracing.
    """
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    
    z_axis = (eye - target) / (np.linalg.norm(eye - target) + 1e-5)
    x_axis = np.cross(up, z_axis) / (np.linalg.norm(np.cross(up, z_axis)) + 1e-5)
    y_axis = np.cross(z_axis, x_axis)
    
    # View Row R
    R = np.eye(4, dtype=np.float32)
    R[0, :3] = x_axis
    R[1, :3] = y_axis
    R[2, :3] = z_axis
    
    # Translation matrix T
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    
    view_matrix = R @ T
    
    inv_view = np.linalg.inv(view_matrix)
    return torch.from_numpy(inv_view).float()

def main():
    print("Starting ray tracing...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"-> Using Device: {device}")

    # Make directory for animation frames
    os.makedirs("anim_frames", exist_ok=True)

    # 1. Initialize model
    max_steps = 64
    shadow_steps = 16
    model = ANERayTracingCore(max_steps=max_steps, shadow_steps=shadow_steps).to(device).half()
    model.eval()

    # 2. Texture input [1, 3, 256, 256]
    dummy_input = torch.zeros(1, 3, 256, 256, dtype=torch.float16, device=device)
    
    y, x = torch.meshgrid(torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256), indexing="ij")
    
    circle_mask = (x*x + y*y) < 0.35
    circle_mask_half = circle_mask.to(device).half()
    
    dummy_input[0, 0, :, :] = circle_mask_half  # Front (X, Y)
    dummy_input[0, 1, :, :] = circle_mask_half  # Top (X, Z)
    dummy_input[0, 2, :, :] = circle_mask_half  # Side (Y, Z)

    # 3. Animation loop (30 frames)
    num_frames = 30
    print(f"Turning camera around in a circular orbit for {num_frames} frames...")
    
    with torch.no_grad():
        for frame in range(num_frames):
          
            angle = (frame / num_frames) * 2.0 * np.pi
            
            # Calculate camera position on circular orbit
            cam_x = 3.5 * np.sin(angle)
            cam_y = 1.5 * np.sin(angle * 0.5) 
            cam_z = 3.5 * np.cos(angle)
            
            inv_view_2d = create_inverse_view_matrix(
                eye=[cam_x, cam_y, cam_z], 
                target=[0.0, 0.0, 0.0], 
                up=[0.0, 1.0, 0.0]
            ).flatten() # 16
            
            # 2. Prepare 64-element zero array for ANE input
            inv_view_64 = torch.zeros(64, dtype=torch.float32)
            inv_view_64[:16] = inv_view_2d
            
            inv_view_4d = inv_view_64.view(1, 64, 1, 1).to(device).half()

          
            output_color = model(dummy_input, inv_view_4d)

            # Save output image
            output_image = output_color.float().cpu()
            output_filename = f"anim_frames/frame_{frame:03d}.png"
            vutils.save_image(output_image, output_filename, normalize=False)
            print(f" Frame {frame+1}/{num_frames} Success -> {output_filename}")

if __name__ == "__main__":
    main()
