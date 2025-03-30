import open3d as o3d
import numpy as np
from data_loader.models import PointCloud

class PointCloudViewer:
    def compare(self, T_gt, T_pred, cloud, save_path=None):
        pcd_gt = self._transform_pcd(cloud.points, T_gt, [1,0,0])
        pcd_pred = self._transform_pcd(cloud.points, T_pred, [0,1,0])
        
        if save_path:
            self._save_visualization(pcd_gt, pcd_pred, save_path)
        o3d.visualization.draw_geometries([pcd_gt, pcd_pred])

    def _transform_pcd(self, points: np.ndarray, T: np.ndarray, color: list):
        """Create colored point cloud from transformation"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            (T[:3,:3] @ points.T).T + T[:3,3]
        )
        pcd.paint_uniform_color(color)
        return pcd