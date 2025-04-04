import os
import numpy as np
from abc import ABC, abstractmethod
import open3d as o3d
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

###############################################################################
# Utility Functions (Sampling, Distances, etc.)
###############################################################################

def sample_points_from_mesh(mesh: o3d.geometry.TriangleMesh, num_points: int = 5000):
    """
    Uniformly sample 'num_points' points on the mesh surface.
    References:
      - Fan et al., "A Point Set Generation Network for 3D Object Reconstruction", CVPR 2017
    """
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    return points, normals

def compute_nearest_distances(ptsA, ptsB):
    """
    For each point in ptsA, compute distance to the nearest neighbor in ptsB.
    We'll use an Open3D KD-tree for speed.
    """
    pcdB = o3d.geometry.PointCloud()
    pcdB.points = o3d.utility.Vector3dVector(ptsB)
    kdtreeB = o3d.geometry.KDTreeFlann(pcdB)

    distances = np.zeros(len(ptsA))
    for i, pt in enumerate(ptsA):
        [_, idx, dist] = kdtreeB.search_knn_vector_3d(pt, 1)
        distances[i] = np.sqrt(dist[0])  # 'dist' is squared distance
    return distances


###############################################################################
# 1. Abstract Interface
###############################################################################

class IEvaluationMetric(ABC):
    """
    Abstract base class for any 3D model evaluation metric.
    Each metric references a well-known paper or standard.
    """
    @abstractmethod
    def evaluate(self, gt_mesh: o3d.geometry.TriangleMesh, 
                 gen_mesh: o3d.geometry.TriangleMesh):
        pass


###############################################################################
# 2. Concrete Metrics
###############################################################################

class ChamferDistanceMetric(IEvaluationMetric):
    """
    Chamfer Distance as per:
      - Fan, Su, & Guibas (CVPR 2017)
    """
    def __init__(self, num_samples=5000):
        self.num_samples = num_samples
    
    def evaluate(self, gt_mesh, gen_mesh):
        # Sample points
        ptsA, _ = sample_points_from_mesh(gt_mesh, self.num_samples)
        ptsB, _ = sample_points_from_mesh(gen_mesh, self.num_samples)

        # Distances A->B
        distAB = compute_nearest_distances(ptsA, ptsB)
        # Distances B->A
        distBA = compute_nearest_distances(ptsB, ptsA)

        # Chamfer = mean(dist(A->B)) + mean(dist(B->A))
        chamfer = distAB.mean() + distBA.mean()
        return chamfer


class HausdorffDistanceMetric(IEvaluationMetric):
    """
    Hausdorff Distance as per:
      - Aspert, Santa-Cruz, & Ebrahimi (ICME 2002)
    """
    def __init__(self, num_samples=5000):
        self.num_samples = num_samples
    
    def evaluate(self, gt_mesh, gen_mesh):
        ptsA, _ = sample_points_from_mesh(gt_mesh, self.num_samples)
        ptsB, _ = sample_points_from_mesh(gen_mesh, self.num_samples)

        distAB = compute_nearest_distances(ptsA, ptsB)
        distBA = compute_nearest_distances(ptsB, ptsA)

        # Hausdorff = max( max(dist(A->B)), max(dist(B->A)) )
        hausdorff = max(distAB.max(), distBA.max())
        return hausdorff


class IoUMetric(IEvaluationMetric):
    """
    3D Intersection-over-Union (IoU) by voxelizing the bounding box.
    Refer to:
      - Wu et al., "3D ShapeNets", CVPR 2015
    This version uses a vectorized grid generation approach to speed up the process.
    """
    def __init__(self, voxel_size=0.01):
        self.voxel_size = voxel_size

    def evaluate(self, gt_mesh, gen_mesh):
        gt_mesh.compute_triangle_normals()
        gen_mesh.compute_triangle_normals()

        # Get bounding boxes for each mesh
        bb_gt = gt_mesh.get_axis_aligned_bounding_box()
        bb_gen = gen_mesh.get_axis_aligned_bounding_box()
        min_bound = np.minimum(bb_gt.min_bound, bb_gen.min_bound)
        max_bound = np.maximum(bb_gt.max_bound, bb_gen.max_bound)

        # Create grid points using a vectorized approach
        xs = np.arange(min_bound[0], max_bound[0], self.voxel_size)
        ys = np.arange(min_bound[1], max_bound[1], self.voxel_size)
        zs = np.arange(min_bound[2], max_bound[2], self.voxel_size)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T.astype(np.float32)
        
        # Optional: Print number of grid points for debugging
        print(f"Generated grid with shape: {grid_points.shape}")

        # Determine which grid points are inside each mesh
        inside_gt = gt_mesh.is_inside(o3d.utility.Vector3dVector(grid_points))
        inside_gen = gen_mesh.is_inside(o3d.utility.Vector3dVector(grid_points))

        inside_gt = np.array(inside_gt, dtype=bool)
        inside_gen = np.array(inside_gen, dtype=bool)

        # Compute intersection and union counts
        intersection = np.logical_and(inside_gt, inside_gen).sum()
        union = np.logical_or(inside_gt, inside_gen).sum()

        iou = intersection / union if union > 0 else 0.0
        return iou



class NormalConsistencyMetric(IEvaluationMetric):
    """
    Normal Consistency as used in:
      - Wang et al., "Pixel2Mesh", ECCV 2018
    Compares surface normals by sampling points and checking the nearest neighbor’s normal.
    """
    def __init__(self, num_samples=5000):
        self.num_samples = num_samples

    def evaluate(self, gt_mesh, gen_mesh):
        ptsA, normalsA = sample_points_from_mesh(gt_mesh, self.num_samples)
        ptsB, normalsB = sample_points_from_mesh(gen_mesh, self.num_samples)

        # KD-tree on B
        pcdB = o3d.geometry.PointCloud()
        pcdB.points = o3d.utility.Vector3dVector(ptsB)
        kdtreeB = o3d.geometry.KDTreeFlann(pcdB)

        dot_products = []
        for i, ptA in enumerate(ptsA):
            nA = normalsA[i]
            [_, idx, dist] = kdtreeB.search_knn_vector_3d(ptA, 1)
            # nearest neighbor index in B
            nn_idx = idx[0]
            nB = normalsB[nn_idx]
            dot = np.dot(nA, nB)
            dot_products.append(dot)
        
        # Average dot product is a measure of normal alignment
        return float(np.mean(dot_products))


class EarthMoversDistanceMetric(IEvaluationMetric):
    """
    Earth Mover's Distance (EMD) as in:
      - Rubner et al., "The Earth Mover’s Distance as a Metric for Image Retrieval", IJCV 2000
      - Fan et al., "A Point Set Generation Network...", CVPR 2017
    We do a discrete approximation with linear assignment.
    """
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples

    def evaluate(self, gt_mesh, gen_mesh):
        ptsA, _ = sample_points_from_mesh(gt_mesh, self.num_samples)
        ptsB, _ = sample_points_from_mesh(gen_mesh, self.num_samples)

        dist_matrix = cdist(ptsA, ptsB, metric='euclidean')
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        emd_value = dist_matrix[row_ind, col_ind].mean()
        return float(emd_value)


###############################################################################
# 3. Evaluation Pipeline
###############################################################################

class ModelEvaluationPipeline:
    """
    Loads ground-truth and generated meshes, then applies 
    a list of evaluation metrics to each pair.
    """
    def __init__(self, metrics):
        """
        :param metrics: A list of metric objects, e.g. [ChamferDistanceMetric(), IoUMetric(), ...]
        """
        self.metrics = metrics

    def load_mesh(self, path: str) -> o3d.geometry.TriangleMesh:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Mesh file not found: {path}")
        mesh = o3d.io.read_triangle_mesh(path)
        if mesh.is_empty():
            raise ValueError(f"Invalid or empty mesh: {path}")
        mesh.compute_vertex_normals()
        return mesh

    def evaluate_pair(self, gt_path: str, gen_path: str):
        """
        :return: dict of {metric_name: score_or_dict_of_scores}
        """
        gt_mesh = self.load_mesh(gt_path)
        gen_mesh = self.load_mesh(gen_path)

        results = {}
        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            score = metric.evaluate(gt_mesh, gen_mesh)
            results[metric_name] = score
        return results


###############################################################################
# 5. Example Usage
###############################################################################

if __name__ == "__main__":
    # Example: create your metrics
    metrics_to_run = [
        #ChamferDistanceMetric(num_samples=2000),
        #HausdorffDistanceMetric(num_samples=2000),
        IoUMetric(voxel_size=0.1),
        #NormalConsistencyMetric(num_samples=2000),
        #EarthMoversDistanceMetric(num_samples=500),
    ]

    # Initialize pipeline
    evaluator = ModelEvaluationPipeline(metrics=metrics_to_run)

    # Example file paths (adapt to your environment)
    gt_path = "models/obj_01.ply"
    gen_path = "genAI_models/obj_01.obj"

    # Evaluate
    results = evaluator.evaluate_pair(gt_path, gen_path)
    
    # Print results
    print("Evaluation Results:")
    for metric_name, score in results.items():
        if isinstance(score, dict):
            print(f"  {metric_name}:")
            for sub_metric, val in score.items():
                print(f"    {sub_metric}: {val:.4f}")
        else:
            print(f"  {metric_name}: {score:.4f}")
