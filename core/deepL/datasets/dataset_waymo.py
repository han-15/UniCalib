import numpy as np
import torch
import glob
from easydict import EasyDict as edict

from .base_dataset import BaseDataset
from .dataset import register_dataset
from ...constant import CameraIntrinsicParameters, EngineMode
from ...utils import read_h5_file, read_image_file


@register_dataset
class DatasetWaymo(BaseDataset):
    def __init__(self, cfg: edict, engine_mode: EngineMode = EngineMode.TRAIN) -> None:
        super(DatasetWaymo, self).__init__(cfg, engine_mode)
        self._depth_images_folder = cfg.depth_imgs_folder
        self._vision_images_folder = cfg.imgs_folder
        self._point_clouds_folder = cfg.pcl_folder
        
        if engine_mode == EngineMode.TRAIN:
            assert 'train_sequences' in cfg, 'train_sequences must be provided in the configuration'
            for sequence in cfg['train_sequences']:
                self.process_sequence(sequence)
        else:
            self.process_sequence(cfg['test_sequence'])
        
        self.test_RT = self.get_test_RT()
   
    def process_sequence(self, sequence: str) -> None:
        """
        Process Waymo sequence.
        
        Args:
            sequence: Sequence identifier.
        """
        lidar_files = sorted(glob.glob(f"{self._cfg['root_folder']}/{sequence}/{self._point_clouds_folder}/*"))
        for lidar_file in lidar_files:
            token = lidar_file.split('/')[-1].split('.')[0]
            data_root = f"{self._cfg['root_folder']}/{sequence}"
            self.all_files.append([token, data_root])

    def _get_test_RT_filename(self) -> str:
        """
        Get test RT filename.
        
        Returns:
            Path to the test RT file.
        """
        return '/'.join([
            self._cfg['root_folder'],
            f'test_RT_seq{self._cfg.test_sequence}_{self._cfg.max_r:.2f}_{self._cfg.max_t:.2f}.csv'
        ])

    def get_camera_parameters(self) -> tuple[CameraIntrinsicParameters, torch.Tensor]:
        """
        Get camera intrinsic parameters.
        
        Returns:
            Tuple of (camera_intrinsic_parameters, camera_extrinsic_parameters).
        """
        camera_intrinsic = CameraIntrinsicParameters(
            2056.282470703125, 2056.282470703125, 939.5780029296875, 641.1030883789062
        )
        return camera_intrinsic, None

    def get_point_cloud_path(self, idx):
        """
        Get point cloud file path.
        
        Args:
            idx: Index of the data sample.
            
        Returns:
            Path to the point cloud file.
        """
        token, data_root = self.all_files[idx]
        return f"{data_root}/{self._point_clouds_folder}/{token}.h5"

    def get_depth_image_path(self, idx):
        """
        Get depth image file path.
        
        Args:
            idx: Index of the data sample.
            
        Returns:
            Path to the depth image file.
        """
        token, data_root = self.all_files[idx]
        return f"{data_root}/{self._depth_images_folder}/{token}/depth_normalized.npy"
    
    def get_image_path(self, idx):
        """
        Get RGB image file path.
        
        Args:
            idx: Index of the data sample.
            
        Returns:
            Path to the RGB image file.
        """
        token, data_root = self.all_files[idx]
        return f"{data_root}/{self._vision_images_folder}/{token}.jpg"

    def __getitem__(self, idx):
        depth_image_path = self.get_depth_image_path(idx)
        try:
            depth_image = np.load(depth_image_path)
            depth_image = np.expand_dims(depth_image, axis=0)
        except IOError as e:
            raise IOError(f"File Broken: {depth_image_path}: {e}")
        
        if self._engine_mode == EngineMode.TEST:
            image_path = self.get_image_path(idx)
            image = read_image_file(image_path)
        
        point_cloud_path = self.get_point_cloud_path(idx)
        point_cloud = self.adjust_point_cloud(read_h5_file(point_cloud_path, 'PC')['PC'])
        
        # Waymo coordinate system adjustment
        point_cloud[1, :] *= -1
        point_cloud[2, :] *= -1
        
        camera_intrinsic_parameters, camera_extrinsic_parameters = self.get_camera_parameters()
        R, T = self.generate_random_transforms(idx)
        
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
