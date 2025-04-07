import logging
import yaml
import cv2
import numpy as np
import trimesh
from typing import List, Dict, Optional
from dataclasses import dataclass

from FoundationPose.Utils import *
from FoundationPose.datareader import *
from FoundationPose.estimater import *
from utils.config import LinemodConfig
from utils.object_mapping import OBJECT_ID_TO_NAME

@dataclass
class PoseEstimationResult:
    video_id: str
    frame_id: str
    object_id: int
    pose: np.ndarray

class LinemodRunner:
    def __init__(self, config: LinemodConfig):
        self.config = config
        self._initialize_output_directory()
        self._setup_logging()
        
    def _initialize_output_directory(self) -> None:
        """Ensure output directory exists"""
        os.makedirs(self.config.DEBUG_DIR, exist_ok=True)

    def _setup_logging(self) -> None:
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.DEBUG_DIR, 'linemod_processing.log')),
                logging.StreamHandler()
            ]
        )

    def _get_object_name(self, ob_id: int) -> str:
        """Get object name with fallback to ID"""
        return OBJECT_ID_TO_NAME.get(ob_id, f"object_{ob_id}")

    def _make_results_yaml_safe(self, results: Dict) -> Dict:
        """Convert numpy arrays and other non-YAML-serializable types to serializable formats"""
        def convert_value(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.int32, np.int64)):
                return int(value)
            if isinstance(value, (np.float32, np.float64)):
                return float(value)
            return value

        safe_results = {}
        for video_id in results:
            safe_results[video_id] = {}
            for frame_id in results[video_id]:
                safe_results[video_id][frame_id] = {}
                for obj_id in results[video_id][frame_id]:
                    safe_results[video_id][frame_id][obj_id] = convert_value(results[video_id][frame_id][obj_id])
        return safe_results

    def _get_mask(self, reader: LinemodReader, frame_idx: int, ob_id: int) -> Optional[np.ndarray]:
        """Get mask for object based on detection type"""
        if self.config.DETECT_TYPE == 'box':
            mask = reader.get_mask(frame_idx, ob_id)
            if mask is None:
                return None
            H, W = mask.shape[:2]
            vs, us = np.where(mask > 0)
            umin, umax = us.min(), us.max()
            vmin, vmax = vs.min(), vs.max()
            valid = np.zeros((H, W), dtype=bool)
            valid[vmin:vmax, umin:umax] = True
            return valid

        elif self.config.DETECT_TYPE == 'mask':
            mask = reader.get_mask(frame_idx, ob_id)
            return mask > 0 if mask is not None else None

        elif self.config.DETECT_TYPE == 'detected':
            mask = cv2.imread(reader.color_files[frame_idx].replace('rgb', 'mask_cosypose'), -1)
            return mask == ob_id if mask is not None else None

        raise ValueError(f"Unknown detection type: {self.config.DETECT_TYPE}")

    def _should_process_object(self, obj_dir: str) -> bool:
        """Determine if we should process this object based on configuration"""
        if self.config.PROCESS_ALL_OBJECTS:
            return True
        return int(obj_dir) in self.config.CUSTOM_OBJECT_IDS

    def _get_objects_to_process(self) -> List[str]:
        """Get list of object directories to process based on configuration"""
        object_dirs = [
            d for d in os.listdir(self.config.LINEMOD_DIR) 
            if os.path.isdir(os.path.join(self.config.LINEMOD_DIR, d)) and d.isdigit()
        ]
        
        if self.config.PROCESS_ALL_OBJECTS:
            return object_dirs
        
        return [f"{ob_id:02d}" for ob_id in self.config.CUSTOM_OBJECT_IDS 
               if f"{ob_id:02d}" in object_dirs]

    def _initialize_estimator(self) -> FoundationPose:
        """Initialize the FoundationPose estimator with a temporary mesh"""
        mesh_tmp = trimesh.primitives.Box(extents=np.ones(3), transform=np.eye(4)).to_mesh()
        glctx = dr.RasterizeCudaContext()
        
        return FoundationPose(
            model_pts=mesh_tmp.vertices.copy(),
            model_normals=mesh_tmp.vertex_normals.copy(),
            symmetry_tfs=None,
            mesh=mesh_tmp,
            scorer=None,
            refiner=None,
            glctx=glctx,
            debug_dir=self.config.DEBUG_DIR,
            debug=self.config.DEBUG_LEVEL
        )

    def _process_object(self, est: FoundationPose, obj_dir: str) -> Dict:
        """Process all frames for a single object"""
        ob_id = int(obj_dir)
        obj_name = self._get_object_name(ob_id)
        logging.info(f"Processing {obj_name} (ID: {ob_id})")
        
        obj_path = os.path.join(self.config.LINEMOD_DIR, obj_dir)
        reader = LinemodReader(obj_path, split=None)
        
        # Load appropriate mesh
        if self.config.USE_RECONSTRUCTED_MESH:
            mesh = reader.get_reconstructed_mesh(ob_id, ref_view_dir=self.config.REF_VIEW_DIR)
        else:
            mesh = reader.get_gt_mesh(ob_id)

        if mesh is None:
            logging.warning(f"Mesh not found for {obj_name}, skipping")
            return {}

        # Configure estimator for this object
        symmetry_tfs = reader.symmetry_tfs.get(ob_id, None)
        est.reset_object(
            model_pts=mesh.vertices.copy(),
            model_normals=mesh.vertex_normals.copy(),
            symmetry_tfs=symmetry_tfs,
            mesh=mesh
        )

        # Prepare output directory
        obj_output_dir = os.path.join(self.config.DEBUG_DIR, obj_name)
        os.makedirs(obj_output_dir, exist_ok=True)

        result = NestDict()
        frame_batch = list(range(len(reader.color_files)))

        for frame_idx in frame_batch:
            try:
                logging.debug(f"Processing frame {frame_idx+1}/{len(frame_batch)} for {obj_name}")

                video_id = reader.get_video_id()
                color = reader.get_color(frame_idx)
                depth = reader.get_depth(frame_idx)
                id_str = reader.id_strs[frame_idx]
                frame_key = str(frame_idx).zfill(6)
                
                if frame_key not in reader.K:
                    logging.warning(f"K matrix not found for frame {frame_key}. Skipping.")
                    result[video_id][id_str][ob_id] = np.eye(4)
                    continue

                K_matrix = np.array(reader.K[frame_key])
                ob_mask = self._get_mask(reader, frame_idx, ob_id)

                if ob_mask is None:
                    logging.warning(f"Mask not found for {obj_name} in frame {frame_idx}. Skipping.")
                    result[video_id][id_str][ob_id] = np.eye(4)
                    continue

                try:
                    est.gt_pose = reader.get_gt_pose(frame_idx, ob_id)
                except:
                    est.gt_pose = None

                pose = est.register(
                    K=K_matrix,
                    rgb=color,
                    depth=depth,
                    ob_mask=ob_mask,
                    ob_id=ob_id
                )

                if self.config.DEBUG_LEVEL >= 3:
                    vis_img = est.last_vis
                    if vis_img is not None:
                        cv2.imwrite(f'{obj_output_dir}/frame_{frame_idx}_vis.png', vis_img)

                result[video_id][id_str][ob_id] = pose

            except Exception as e:
                logging.error(f"Error processing frame {frame_idx} for {obj_name}: {str(e)}")
                if video_id and id_str:
                    result[video_id][id_str][ob_id] = np.eye(4)
                continue

        # Save results for this object
        with open(f'{obj_output_dir}/linemod_res.yml', 'w') as ff:
            safe_result = self._make_results_yaml_safe(result)
            yaml.safe_dump(safe_result, ff)

        return result

    def run(self) -> None:
        """Main execution method to process all specified objects"""
        wp.force_load(device=self.config.DEVICE)
        combined_res = NestDict()
        est = self._initialize_estimator()

        objects_to_process = self._get_objects_to_process()
        logging.info(f"Processing objects: {objects_to_process}")

        for obj_dir in objects_to_process:
            try:
                result = self._process_object(est, obj_dir)
                # Merge results
                for video_id in result:
                    for id_str in result[video_id]:
                        for ob_id in result[video_id][id_str]:
                            combined_res[video_id][id_str][ob_id] = result[video_id][id_str][ob_id]

            except Exception as e:
                obj_name = self._get_object_name(int(obj_dir))
                logging.error(f"Error processing {obj_name} (ID: {obj_dir}): {str(e)}")
                continue

        # Save combined results
        if combined_res:
            with open(f'{self.config.DEBUG_DIR}/linemod_res_combined.yml', 'w') as ff:
                safe_combined_res = self._make_results_yaml_safe(combined_res)
                yaml.safe_dump(safe_combined_res, ff)

        logging.info(f"Pose estimation completed. Results saved in {self.config.DEBUG_DIR}")