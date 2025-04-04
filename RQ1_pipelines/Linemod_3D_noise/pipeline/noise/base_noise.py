from abc import ABC, abstractmethod
import open3d as o3d

class BaseNoise(ABC):
    @abstractmethod
    def apply(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        pass
