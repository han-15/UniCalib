import os
import numpy as np
import torch
import pykitti
from easydict import EasyDict as edict

from .base_kitti_dataset import BaseKittiDataset
from .dataset import register_dataset
from ...constant import CameraIntrinsicParameters, EngineMode
from ...utils import is_path_exist, read_image_file


@register_dataset
class DatasetKITTIRaw(BaseKittiDataset):
    def __init__(self, cfg: edict, engine_mode: EngineMode = EngineMode.TRAIN) -> None:
        super(DatasetKITTIRaw, self).__init__(cfg, engine_mode)
        self.train_dataset = []
        
        if engine_mode == EngineMode.TRAIN:
            assert 'train_sequences' in cfg, 'train_sequences must be provided in the configuration'
            self.date = cfg['train_date']
            for sequence in cfg['train_sequences']:
                self.process_sequence(sequence)
                self.train_dataset.append(pykitti.raw(self._cfg['root_folder'], self.date, sequence))
                velo2cam2 = torch.from_numpy(self.train_dataset[-1].calib.T_cam2_velo).float()
                sequence = self.date + '_drive_' + sequence + '_sync'
                self.T_cam2_velo[sequence] = velo2cam2
                velo2cam3 = torch.from_numpy(self.train_dataset[-1].calib.T_cam3_velo).float()
                self.T_cam3_velo[sequence] = velo2cam3
        else:
            self.date = cfg['test_date']
            for sequence in cfg['test_sequences']:
                self.process_sequence(sequence)
                self.test_dataset = pykitti.raw(self._cfg['root_folder'], self.date, sequence)
                velo2cam2 = torch.from_numpy(self.test_dataset.calib.T_cam2_velo).float()
                sequence = self.date + '_drive_' + sequence + '_sync'
                self.T_cam2_velo[sequence] = velo2cam2
                velo2cam3 = torch.from_numpy(self.test_dataset.calib.T_cam3_velo).float()
                self.T_cam3_velo[sequence] = velo2cam3

        self.test_RT = self.get_test_RT()
    
    def process_sequence(self, sequence: str) -> None:
        """
        Process KITTI Raw sequence.
        
        Args:
            sequence: Sequence identifier.
        """
        sequence = self.date + '_drive_' + sequence + '_sync'
        image_files = sorted(os.listdir(
            os.path.join(self._cfg['root_folder'], self.date, sequence, 'image_02', 'data')
        ))
        for image_file in image_files:
            timestamp_formatted = image_file.split('.')[0]
            map_file_path = [self._cfg['root_folder'], self.date, sequence, 
                           self._cfg['pcl_folder'], 'data', timestamp_formatted + '.bin']
            img_file_path = [self._cfg['root_folder'], self.date, sequence, 'image_02',
                           self._cfg['imgs_folder'], timestamp_formatted, 'depth_normalized.npy']
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
            self._cfg['root_folder'], self.date,
            f'test_RT_{self._cfg.test_date}_{self._cfg.max_r:.2f}_{self._cfg.max_t:.2f}.csv'
        ])

    def get_camera_parameters(self) -> tuple[CameraIntrinsicParameters, torch.Tensor]:
        """
        Get camera intrinsic parameters.
        
        Returns:
            Tuple of (camera_intrinsic_parameters, camera_extrinsic_parameters).
        """
        if self.date == '2011_09_26':
            camera_intrinsic_parameters = CameraIntrinsicParameters(721.5377, 721.5377, 609.5593, 172.854)
        elif self.date == '2011_09_30':
            camera_intrinsic_parameters = CameraIntrinsicParameters(707.0912, 707.0912, 601.8873, 183.1104)
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
            self._cfg['root_folder'], self.date, sequence, 
            self._cfg['pcl_folder'], 'data', timestamp + '.bin'
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
            self._cfg['root_folder'], self.date, sequence, 'image_02',
            self._cfg['imgs_folder'], timestamp, 'depth_normalized.npy'
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
            self._cfg['root_folder'], self.date, sequence, 'image_02', 'data', timestamp + '.png'
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
        # print("R and T shape:")
        # print(R.shape, T.shape)
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
