import os
import numpy as np
import torch
from easydict import EasyDict as edict

from .base_kitti_dataset import BaseKittiDataset
from .dataset import register_dataset
from ...constant import CameraIntrinsicParameters, EngineMode
from ...utils import is_path_exist, read_image_file


@register_dataset
class DatasetKITTI360(BaseKittiDataset):
    def __init__(self, cfg: edict, engine_mode: EngineMode = EngineMode.TRAIN) -> None:
        super(DatasetKITTI360, self).__init__(cfg, engine_mode)
        
        velo2cam = np.array([
            [0.04307104361409124, -0.9990043710100727, -0.011625485582612508, 0.26234696454785045],
            [-0.0882928649736657, 0.0077846140388720024, -0.9960641393796895, -0.10763413732850423],
            [0.9951629289402036, 0.04392796941355016, -0.08786966658919294, -0.8292052503750126],
            [0, 0, 0, 1]
        ])
        self.velo2cam = torch.from_numpy(velo2cam).float()
        
        if engine_mode != EngineMode.TRAIN:
            self.process_sequence(self._cfg['test_sequence'])
        
        self.test_RT = self.get_test_RT()

    def process_sequence(self, sequence: str) -> None:
        """
        Process KITTI-360 sequence.
        
        Args:
            sequence: Sequence identifier.
        """
        image_files = sorted(os.listdir(
            os.path.join(self._cfg['root_folder'], sequence, 'image_00', 'data_rect')
        ))
        for image_file in image_files:
            timestamp_formatted = image_file.split('.')[0]
            map_file_path = [self._cfg['root_folder'], sequence, self._cfg['pcl_folder'], 
                           timestamp_formatted + '.bin']
            img_file_path = [self._cfg['root_folder'], sequence, self._cfg['depth_imgs_folder'], 
                           timestamp_formatted, 'depth_normalized.npy']
            if not (is_path_exist(*map_file_path) and is_path_exist(*img_file_path)):
                continue
            self.all_files.append('/'.join([sequence, timestamp_formatted]))

    def _get_test_RT_filename(self) -> str:
        """
        Get test RT filename.
        
        Returns:
            Path to the test RT file.
        """
        return '/'.join([
            self._cfg['root_folder'],
            f'test_RT_{self._cfg.test_sequence}_{self._cfg.max_r:.2f}_{self._cfg.max_t:.2f}.csv'
        ])

    def adjust_kitti_point_cloud(self, pc: np.ndarray, sequence) -> torch.Tensor:
        """
        Preprocess KITTI-360 point cloud and transform to camera coordinate system.
        
        Args:
            pc: Point cloud array.
            sequence: Sequence identifier.
            
        Returns:
            Point cloud in camera coordinates with shape (4, N).
        """
        pc_in = self.adjust_point_cloud(pc)
        pc_in = torch.mm(self.velo2cam, pc_in)
        pc_in = pc_in[[2, 0, 1, 3], :]
        return pc_in

    def get_camera_parameters(self) -> tuple[CameraIntrinsicParameters, torch.Tensor]:
        """
        Get camera intrinsic parameters.
        
        Returns:
            Tuple of (camera_intrinsic_parameters, camera_extrinsic_parameters).
        """
        camera_intrinsic_parameters = CameraIntrinsicParameters(552.554261, 552.554261, 682.049453, 238.769549)
        return camera_intrinsic_parameters, None

    def get_point_cloud_path(self, idx):
        """
        Get point cloud file path and sequence identifier.
        
        Args:
            idx: Index of the data sample.
            
        Returns:
            Tuple of (point_cloud_path, sequence).
        """
        item = self.all_files[idx]
        sequence = str(item.split('/')[0])
        timestamp = str(item.split('/')[1])
        pointcloud_path = os.path.join(
            self._cfg['root_folder'], sequence, 'velodyne_points', 'data', timestamp + '.bin'
        )
        return pointcloud_path, sequence

    def get_depth_image_path(self, idx) -> str:
        """
        Get depth image file path.
        
        Args:
            idx: Index of the data sample.
            
        Returns:
            Path to the depth image file.
        """
        item = self.all_files[idx]
        sequence = str(item.split('/')[0])
        timestamp = str(item.split('/')[1])
        image_path = os.path.join(
            self._cfg['root_folder'], sequence, self._cfg['depth_imgs_folder'],
            timestamp, 'depth_normalized.npy'
        )
        return image_path

    def get_image_path(self, idx) -> str:
        """
        Get RGB image file path.
        
        Args:
            idx: Index of the data sample.
            
        Returns:
            Path to the RGB image file.
        """
        item = self.all_files[idx]
        sequence = str(item.split('/')[0])
        timestamp = str(item.split('/')[1])
        image_path = os.path.join(
            self._cfg['root_folder'], sequence, self._cfg['imgs_folder'], timestamp + '.png'
        )
        return image_path

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

        point_cloud_path, sequence = self.get_point_cloud_path(idx)
        point_cloud = self.adjust_kitti_point_cloud(
            self.load_velo_scan(point_cloud_path)[:, :3], sequence
        )

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
