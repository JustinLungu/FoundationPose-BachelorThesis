import open3d as o3d
import numpy as np

# Create and sample point cloud
mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
mesh.compute_vertex_normals()
pcd = mesh.sample_points_poisson_disk(500)
pcd_np = np.asarray(pcd.points).astype(np.float32)

# Create CUDA device object explicitly
device = o3d.core.Device("CUDA:0")

# Create GPU tensor point cloud
pcd_tensor = o3d.t.geometry.PointCloud()
pcd_tensor.point["positions"] = o3d.core.Tensor(
    pcd_np, dtype=o3d.core.Dtype.Float32, device=device
)

# Apply safe transform (instead of translate)
T = o3d.core.Tensor.eye(4, dtype=o3d.core.Dtype.Float32, device=device)
T[0, 3] = 0.5
pcd_tensor_translated = pcd_tensor.clone().transform(T)

# Convert back to legacy CPU point clouds
pcd1_legacy = pcd_tensor.to_legacy()
pcd2_legacy = pcd_tensor_translated.to_legacy()

# Save to disk
o3d.io.write_point_cloud("pcd1_cuda.ply", pcd1_legacy)
o3d.io.write_point_cloud("pcd2_cuda.ply", pcd2_legacy)

print("✅ Success: Saved GPU-based point clouds.")
