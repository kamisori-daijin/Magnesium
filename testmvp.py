import torch
import numpy as np
import matplotlib.pyplot as plt

from MVPProcessor import ANEMVPProcessor


NUM_VERTICES = 10000

# 1. Initialize model
model = ANEMVPProcessor(max_vertices=NUM_VERTICES)
model.eval()

def create_camera_matrix(eye, target, up):
    """
    eye: Camera Position
    target: Target Position
    up: Camera Up
    """
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    
    # Eye Direction
    z_axis = (eye - target)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # right direction
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # up direction
    y_axis = np.cross(z_axis, x_axis)
    
    # rotation matrix
    R = np.eye(4, dtype=np.float32)
    R[0, :3] = x_axis
    R[1, :3] = y_axis
    R[2, :3] = z_axis
    
    # translation matrix
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    
    # view matrix = R * T
    matrix = torch.from_numpy(R @ T)
    return matrix

# calculate camera position
distance = 3.5
yaw = np.radians(45.0)
pitch = np.radians(30.0)

cam_x = distance * np.cos(pitch) * np.sin(yaw)
cam_y = distance * np.sin(pitch)
cam_z = distance * np.cos(pitch) * np.cos(yaw)

# set camera position
camera_mat = create_camera_matrix(
    eye=[cam_x, cam_y, cam_z], 
    target=[0.0, 0.0, 0.0], 
    up=[0.0, 1.0, 0.0]
)


np.random.seed(42)
raw_xyz = np.random.uniform(-0.8, 0.8, (3, NUM_VERTICES)).astype(np.float32)

raw_w = np.ones((1, NUM_VERTICES), dtype=np.float32)
vertex_buffer_np = np.vstack([raw_xyz, raw_w])[np.newaxis, ...] # (1, 4, 10000)
vertex_buffer = torch.from_numpy(vertex_buffer_np)


with torch.no_grad():
    output_buffer = model(camera_mat, vertex_buffer) # (1, 3, 10000)

# visualize results
transformed_points = output_buffer.squeeze(0).numpy()

screen_x = transformed_points[0] # 0ch is already X
screen_y = transformed_points[1] # 1ch is already Y
depth = transformed_points[2]    # 2ch is already Z

# Plot the results
plt.figure(figsize=(6, 6))

plt.scatter(screen_x, screen_y, s=1, c=depth, cmap='viridis', alpha=0.6)
plt.title(f"ANE MVP Processor Test ({NUM_VERTICES} Vertices)")
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')

output_png = "mvp_processor_test.png"
plt.savefig(output_png, dpi=150)
plt.close()

print(f"Success! Check {output_png}")
