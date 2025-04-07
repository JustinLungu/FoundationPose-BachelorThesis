import numpy as np
import trimesh
import imageio
import cv2
import glob
import logging
from typing import Optional, List
from dataclasses import dataclass

from FoundationPose.estimater import *
from FoundationPose.datareader import *
from utils.config import DemoConfig

@dataclass
class PoseEstimationResult:
    pose: np.ndarray
    visualization: np.ndarray

class DemoRunner:
    def __init__(self, config: DemoConfig):
        self.config = config
        self._initialize_output_directory()
        self._setup_logging()
        
    def _initialize_output_directory(self) -> None:
        """Ensure output directory exists"""
        os.makedirs(self.config.OUTPUT_ROOT, exist_ok=True)

    def _setup_logging(self) -> None:
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.OUTPUT_ROOT, 'demo_processing.log')),
                logging.StreamHandler()
            ]
        )

    def _should_process_object(self, obj_name: str) -> bool:
        """Determine if we should process this object based on configuration"""
        if self.config.PROCESS_ALL_OBJECTS:
            return True
        return obj_name in self.config.CUSTOM_OBJECT_IDS

    def _validate_object_data(self, data_root: str) -> bool:
        """Check if object has all required data"""
        mesh_file = os.path.join(data_root, "mesh", "model.obj")
        rgb_files = glob.glob(os.path.join(data_root, "rgb", "*.png"))
        
        if not os.path.exists(mesh_file):
            logging.warning(f"Mesh file not found at {mesh_file}")
            return False
        if len(rgb_files) == 0:
            logging.warning(f"No RGB images found in {os.path.join(data_root, 'rgb')}")
            return False
            
        # Check if depth images exist for all RGB images
        for rgb_file in rgb_files:
            depth_file = rgb_file.replace('rgb', 'depth')
            if not os.path.exists(depth_file):
                logging.warning(f"Depth file not found for {rgb_file}")
                return False
                
        return True

    def _setup_object_directories(self, obj_name: str) -> str:
        """Create output directories for this object"""
        debug_dir = os.path.join(self.config.OUTPUT_ROOT, obj_name)
        os.makedirs(debug_dir, exist_ok=True)
        os.makedirs(os.path.join(debug_dir, "ob_in_cam"), exist_ok=True)
        os.makedirs(os.path.join(debug_dir, "track_vis"), exist_ok=True)
        return debug_dir

    def _initialize_pose_estimator(self, mesh: trimesh.Trimesh, debug_dir: str) -> FoundationPose:
        """Initialize the FoundationPose estimator"""
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()

        return FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=debug_dir,
            debug=self.config.DEBUG_LEVEL,
            glctx=glctx
        )

    def _process_frame(self, est: FoundationPose, reader: YcbineoatReader, 
                      frame_idx: int, to_origin: np.ndarray, bbox: np.ndarray) -> Optional[PoseEstimationResult]:
        """Process a single frame and return pose estimation results"""
        try:
            color = reader.get_color(frame_idx)
            depth = reader.get_depth(frame_idx)
            
            if depth is None:
                logging.error(f"Failed to load depth image for frame {frame_idx}")
                return None

            if frame_idx == 0 or self.config.USE_MASK_EVERY_FRAME:
                mask = reader.get_mask(frame_idx).astype(bool)
                pose = est.register(
                    K=reader.K, 
                    rgb=color, 
                    depth=depth, 
                    ob_mask=mask, 
                    iteration=self.config.ITERATION_REGISTER
                )
            else:
                pose = est.track_one(
                    rgb=color, 
                    depth=depth, 
                    K=reader.K, 
                    iteration=self.config.ITERATION_TRACK
                )

            # Create visualization
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(
                color, 
                ob_in_cam=center_pose, 
                scale=self.config.AXIS_SCALE, 
                K=reader.K, 
                thickness=self.config.AXIS_THICKNESS, 
                transparency=self.config.TRANSPARENCY, 
                is_input_rgb=True
            )

            return PoseEstimationResult(pose=pose, visualization=vis)
            
        except Exception as e:
            logging.error(f"Error processing frame {frame_idx}: {str(e)}")
            return None

    def _save_results(self, debug_dir: str, result: PoseEstimationResult, frame_id: str) -> None:
        """Save pose and visualization results"""
        try:
            np.savetxt(
                os.path.join(debug_dir, "ob_in_cam", f"{frame_id}.txt"), 
                result.pose.reshape(4, 4)
            )
            imageio.imwrite(
                os.path.join(debug_dir, "track_vis", f"{frame_id}.png"), 
                result.visualization
            )
        except Exception as e:
            logging.error(f"Failed to save results for frame {frame_id}: {str(e)}")

    def run(self) -> None:
        """Main execution method to process all specified objects"""
        set_logging_format()
        set_seed(0)

        for obj_name in sorted(os.listdir(self.config.DEMO_ROOT)):
            if not self._should_process_object(obj_name):
                logging.info(f"Skipping '{obj_name}' (not in processing list)")
                continue

            data_root = os.path.join(self.config.DEMO_ROOT, obj_name)
            if not self._validate_object_data(data_root):
                logging.warning(f"Skipping '{obj_name}' (missing data)")
                continue

            logging.info(f"\nProcessing object: {obj_name}")
            debug_dir = self._setup_object_directories(obj_name)
            
            try:
                # Load mesh and initialize estimator
                mesh = trimesh.load(os.path.join(data_root, "mesh", "model.obj"))
                to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
                bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
                
                est = self._initialize_pose_estimator(mesh, debug_dir)
                reader = YcbineoatReader(
                    video_dir=data_root, 
                    shorter_side=None, 
                    zfar=np.inf, 
                    per_frame_masks=True
                )

                # Process all frames
                for frame_idx in range(len(reader.color_files)):
                    logging.info(f'Processing frame {frame_idx}')
                    result = self._process_frame(est, reader, frame_idx, to_origin, bbox)
                    
                    if result is None:
                        continue
                        
                    self._save_results(debug_dir, result, reader.id_strs[frame_idx])

                    # Optional display
                    cv2.imshow("Prediction", result.visualization[..., ::-1])
                    if cv2.waitKey(1) == ord('q'):
                        break
                        
            except Exception as e:
                logging.error(f"Error processing object {obj_name}: {str(e)}")
                continue