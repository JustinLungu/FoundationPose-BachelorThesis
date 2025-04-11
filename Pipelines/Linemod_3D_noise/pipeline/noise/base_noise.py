from abc import ABC, abstractmethod
import open3d as o3d

class BaseNoise(ABC):
    @abstractmethod
    def apply(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Apply noise to a mesh and return the modified version.
        Args:
            mesh: Input triangle mesh to corrupt
        Returns:
            Modified mesh with noise applied
        """
        pass
