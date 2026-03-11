import os
import numpy as np
from easydict import EasyDict as edict

from .base_dataset import BaseDataset
from .dataset import register_dataset
from ...constant import CameraIntrinsicParameters, EngineMode
from ...utils import is_path_exist, read_image_file, get_logger, read_matrix_txt_file


@register_dataset
class DatasetCustomTest(BaseDataset):
    """
    Custom test dataset class that automatically scans and loads test data from the images folder.
    
    Configuration parameters (cfg):
        - root_folder: Root directory path
        - pcl_folder: Point cloud folder name (relative to root_folder)
        - imgs_folder: RGB image folder name (relative to root_folder)
        - depth_imgs_folder: Depth image folder name (relative to root_folder)
        - w_scale: Width scale ratio
        - h_scale: Height scale ratio
        - adjust_coordinate_order: Coordinate order adjustment (optional)
        - rotate_lidar_deg: LiDAR rotation angle around Z-axis (degrees)
        - scale_lidar: LiDAR point cloud scale factor
        - translate_lidar_z: LiDAR translation along Z-axis (meters)
    
    Constructor parameters:
        - camera_intrinsic: CameraIntrinsicParameters object, camera intrinsics
        - Other user-defined parameters
    
    Directory structure example:
        root_folder/
        ├── images/              # RGB image folder
        │   ├── 000000.jpg
        │   ├── 000001.jpg
        │   └── ...
        ├── depth_images/        # Depth image folder
        │   ├── 000000.npy
        │   ├── 000001.npy
        │   └── ...
        └── pointclouds/         # Point cloud folder
            ├── 000000.bin
            ├── 000001.bin
            └── ...
    """
    
    def __init__(self, cfg: edict, camera_intrinsic: CameraIntrinsicParameters, 
                 engine_mode: EngineMode = EngineMode.TEST, **kwargs):
        """
        Initialize custom test dataset.
        
        Args:
            cfg: Configuration dictionary containing folder paths and other parameters
            camera_intrinsic: Camera intrinsic parameters object
            engine_mode: Engine mode (defaults to TEST)
            **kwargs: Other user-defined parameters
        """
        super(DatasetCustomTest, self).__init__(cfg, engine_mode)
        
        # Save configuration parameters
        self.root_folder = cfg.root_folder
        self.pcl_folder = cfg.pcl_folder
        self.imgs_folder = cfg.imgs_folder
        self.depth_imgs_folder = cfg.depth_imgs_folder
        self.camera_intrinsic = camera_intrinsic
        self.extrinsics_folder = cfg.extrinsics_folder
        self.lidar_poses_folder = cfg.lidar_poses_folder
        
        # LiDAR sensor coordinate system transformation parameters
        self.rotate_lidar_deg = cfg.get('rotate_lidar_deg', 0.0)
        self.scale_lidar = cfg.get('scale_lidar', 1.0)
        self.translate_lidar_z = cfg.get('translate_lidar_z', 0.0)
        
        # Save user-defined parameters
        self.user_params = kwargs
        
        # Scan images folder and build file list
        self.process_images_folder()
        
        # Get test RT
        self.test_RT = self.get_test_RT()
        
        get_logger().success(f"Successfully loaded {len(self.all_files)} data samples")
        get_logger().info(f"LiDAR transform parameters: rotation={self.rotate_lidar_deg}deg, scale={self.scale_lidar}, Z-translation={self.translate_lidar_z}m")
    
    def process_images_folder(self):
        """
        Scan the images folder and match corresponding depth images and point cloud files by image filename.
        """
        imgs_path = os.path.join(self.root_folder, self.imgs_folder)
        
        if not is_path_exist(imgs_path):
            raise FileNotFoundError(f"Image folder does not exist: {imgs_path}")
        
        get_logger().info(f"Scanning image folder: {imgs_path}")
        
        # Get all image files and sort them
        image_files = sorted([f for f in os.listdir(imgs_path) 
                            if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        for image_file in image_files:
            # Get filename (without extension)
            filename = os.path.splitext(image_file)[0]
            
            # Build corresponding depth image and point cloud file paths
            depth_file_path = os.path.join(self.root_folder, self.depth_imgs_folder, 
                                          filename + '/depth_normalized.npy')
            pcl_file_path = os.path.join(self.root_folder, self.pcl_folder, 
                                        filename.split('_')[0] + '.bin')
            
            # Check if files exist
            if not is_path_exist(depth_file_path):
                get_logger().info(f"Depth image does not exist, skipping: {depth_file_path}")
                continue
            if not is_path_exist(pcl_file_path):
                get_logger().info(f"Point cloud file does not exist, skipping: {pcl_file_path}")
                continue
            
            # Add to file list
            self.all_files.append(filename)
    
    def process_sequence(self, sequence: str) -> None:
        """
        Process sequence (this dataset does not need this method, kept to satisfy base class requirement).
        
        Args:
            sequence: Sequence identifier
        """
        pass
    
    def _get_test_RT_filename(self) -> str:
        """
        Get test RT filename (used for saving random transformations).
        
        Returns:
            Test RT file path
        """
        return os.path.join(
            self.root_folder,
            f'test_RT_custom_{self._cfg.max_r:.2f}_{self._cfg.max_t:.2f}.csv'
        )
    
    def get_camera_parameters(self, idx: int) -> tuple[CameraIntrinsicParameters, None]:
        """
        Get camera parameters.
        
        Args:
            idx: Data sample index
            
        Returns:
            Tuple (camera intrinsics, None), extrinsics is None because the test dataset does not need it
        """
        # Return the camera intrinsics passed in the constructor
        # return CameraIntrinsicParameters(
        #     self.camera_intrinsic.focal_length_x,
        #     self.camera_intrinsic.focal_length_y,
        #     self.camera_intrinsic.principal_point_x,
        #     self.camera_intrinsic.principal_point_y
        # ), None
        # Hardcoded camera intrinsics
        return CameraIntrinsicParameters(
            3665.70215000,  # fx
            3665.74219000,  # fy
            963.83746500,   # cx
            543.07129000    # cy
        ), None

    
    def get_point_cloud_path(self, idx: int) -> str:
        """
        Get point cloud file path.
        
        Args:
            idx: Data sample index
            
        Returns:
            Point cloud file path
        """
        filename = self.all_files[idx]
        # pcl_path = os.path.join(self.root_folder, self.pcl_folder, filename + '.bin')
        pcl_path = os.path.join(self.root_folder, self.pcl_folder, 
                                        filename.split('_')[0] + '.bin')
        return pcl_path
    
    def get_depth_image_path(self, idx: int) -> str:
        """
        Get depth image file path.
        
        Args:
            idx: Data sample index
            
        Returns:
            Depth image file path
        """
        filename = self.all_files[idx]
        depth_path = os.path.join(self.root_folder, self.depth_imgs_folder, filename + '/depth_normalized_log.npy')
        # depth_path = os.path.join(self.root_folder, self.depth_imgs_folder, filename + '/depth_normalized.npy')
        return depth_path
    
    def get_image_path(self, idx: int) -> str:
        """
        Get RGB image file path.
        
        Args:
            idx: Data sample index
            
        Returns:
            RGB image file path
        """
        filename = self.all_files[idx]
        # Try different image extensions
        for ext in ['.jpg', '.png', '.jpeg']:
            img_path = os.path.join(self.root_folder, self.imgs_folder, filename + ext)
            if is_path_exist(img_path):
                return img_path
        # If none exist, return the default jpg path (will error later)
        return os.path.join(self.root_folder, self.imgs_folder, filename + '.jpg')
    def get_extrinsics(self, idx: int):
        """
        Get camera extrinsics (cam_pose: camera pose in world coordinate system).
        
        Args:
            idx: Data sample index
            
        Returns:
            np.ndarray: 4x4 transformation matrix, or None if it does not exist
        """
        filename = self.all_files[idx]
        extrinsics_path = os.path.join(self.root_folder, self.extrinsics_folder, filename + '.txt')
        if is_path_exist(extrinsics_path):
            return read_matrix_txt_file(extrinsics_path)
        return None

    def get_lidar_pose(self, idx: int):
        """
        Get LiDAR pose (lidar_pose: LiDAR pose in world coordinate system).
        
        Args:
            idx: Data sample index
            
        Returns:
            np.ndarray: 4x4 transformation matrix, or None if it does not exist
        """
        filename = self.all_files[idx]
        lidar_pose_path = os.path.join(self.root_folder, self.lidar_poses_folder, filename.split('_')[0] + '.txt')
        if is_path_exist(lidar_pose_path):
            return read_matrix_txt_file(lidar_pose_path)
        return None
    
    def rotate_z(self, points: np.ndarray, deg: float) -> np.ndarray:
        """
        Rotate point cloud around Z-axis.
        
        Args:
            points: Point cloud array, shape (N, 3)
            deg: Rotation angle (degrees)
            
        Returns:
            Rotated point cloud array, shape (N, 3)
        """
        if deg == 0.0:
            return points
        
        theta = np.deg2rad(deg)
        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return (rot @ points.T).T
    
    def load_point_cloud(self, pcl_path: str) -> np.ndarray:
        """
        Load point cloud file and apply sensor coordinate system transformations. Supports .bin and .npy formats.
        
        Args:
            pcl_path: Point cloud file path
            
        Returns:
            Transformed point cloud array, shape (N, 3) or (N, 4)
        """
        if pcl_path.endswith('.bin'):
            # KITTI format binary point cloud file
            point_cloud = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 4)
        elif pcl_path.endswith('.npy'):
            # NumPy format point cloud file
            point_cloud = np.load(pcl_path)
        else:
            raise ValueError(f"Unsupported point cloud file format: {pcl_path}")
        
        # Apply sensor coordinate system transformations (consistent with mb_project_lidar.py)
        # Extract XYZ coordinates
        pts = point_cloud[:, :3].astype(np.float64)
        
        # 1. Scale
        pts *= self.scale_lidar
        
        # 2. Z-axis translation
        pts[:, 2] += self.translate_lidar_z
        
        # 3. Rotate around Z-axis
        pts = self.rotate_z(pts, self.rotate_lidar_deg)
        
        # If the original point cloud has intensity information, preserve it
        if point_cloud.shape[1] == 4:
            point_cloud = np.concatenate([pts, point_cloud[:, 3:4]], axis=1)
        else:
            point_cloud = pts
        
        return point_cloud
    
    def __getitem__(self, idx: int):
        """
        Get a data sample.
        
        Args:
            idx: Data sample index
            
        Returns:
            Dictionary containing the following keys:
                - vision_image: Depth image tensor
                - original_image: Original RGB image tensor
                - point_cloud: Point cloud tensor
                - camera_intrinsic_parameters: Camera intrinsics
                - tr_error: Translation error
                - rot_error: Rotation error
                - order: Coordinate order
        """
        # Load depth image
        depth_image_path = self.get_depth_image_path(idx)
        try:
            if depth_image_path.endswith('.npy'):
                depth_image = np.load(depth_image_path)
            elif depth_image_path.endswith('.png') or depth_image_path.endswith('.jpg'):
                depth_image = read_image_file(depth_image_path)
                if len(depth_image.shape) == 3:
                    depth_image = depth_image[:, :, 0]  # Take the first channel
            else:
                raise ValueError(f"Unsupported depth image format: {depth_image_path}")
            
            # Ensure the depth image is 3-dimensional (1, H, W)
            if len(depth_image.shape) == 2:
                depth_image = np.expand_dims(depth_image, axis=0)
        except IOError as e:
            raise IOError(f"Depth image file is corrupted: {depth_image_path}: {e}")
        
        # Load RGB image (only in TEST mode)
        if self._engine_mode == EngineMode.TEST:
            image_path = self.get_image_path(idx)
            image = read_image_file(image_path)
        
        # Load point cloud (with sensor coordinate system transforms applied: scale, translate, rotate)
        point_cloud_path = self.get_point_cloud_path(idx)
        point_cloud = self.load_point_cloud(point_cloud_path)
        
        # Only take XYZ coordinates (ignore intensity information if present)
        if point_cloud.shape[1] >= 3:
            point_cloud = point_cloud[:, :3]
        
        # Apply lidar_pose transform: from LiDAR coordinate system to world coordinate system
        lidar_pose = self.get_lidar_pose(idx)
        if lidar_pose is not None:
            # Convert to homogeneous coordinates (N, 4)
            pts_h = np.concatenate([point_cloud, np.ones((point_cloud.shape[0], 1), dtype=np.float64)], axis=1)
            # Apply lidar_pose transform
            world_pts = (lidar_pose @ pts_h.T).T[:, :3]
            point_cloud = world_pts
        
        # Apply extrinsics transform: from world coordinate system to camera coordinate system
        extrinsics = self.get_extrinsics(idx)
        if extrinsics is not None:
            # Convert to homogeneous coordinates (N, 4)
            pts_h = np.concatenate([point_cloud, np.ones((point_cloud.shape[0], 1), dtype=np.float64)], axis=1)
            # extrinsics is cam_pose, need to invert to get world_to_cam
            world_to_cam = np.linalg.inv(extrinsics)
            # Apply world_to_cam transform
            cam_pts = (world_to_cam @ pts_h.T).T[:, :3]
            point_cloud = cam_pts
        
        # Adjust point cloud format to (4, N)
        point_cloud = self.adjust_point_cloud(point_cloud)
        
        # Get camera parameters
        camera_intrinsic_parameters, camera_extrinsic_parameters = self.get_camera_parameters(idx)
        
        R, T = self.generate_zero_transforms(idx)
        if self._engine_mode != EngineMode.TEST:
            depth_image, point_cloud, camera_intrinsic_parameters = self.augment_data(
                depth_image, point_cloud, camera_intrinsic_parameters, camera_extrinsic_parameters
            )
            return {
                'vision_image': depth_image,
                'point_cloud': point_cloud,
                'camera_intrinsic_parameters': camera_intrinsic_parameters,
                'tr_error': T,
                'rot_error': R,
                'order': self._adjust_coordinate_order
            }
        else:
            depth_image, image, point_cloud, camera_intrinsic_parameters = self.augment_data_all(
                depth_image, image, point_cloud, camera_intrinsic_parameters, camera_extrinsic_parameters
            )
            return {
                'vision_image': depth_image,
                'original_image': image,
                'point_cloud': point_cloud,
                'camera_intrinsic_parameters': camera_intrinsic_parameters,
                'tr_error': T,
                'rot_error': R,
                'order': self._adjust_coordinate_order
            }
