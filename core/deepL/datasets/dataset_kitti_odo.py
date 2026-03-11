import os
import numpy as np
import torch
import pykitti
from easydict import EasyDict as edict

from .base_kitti_dataset import BaseKittiDataset
from .dataset import register_dataset
from ...constant import CameraIntrinsicParameters, EngineMode
from ...utils import is_path_exist, read_txt_file, read_image_file


@register_dataset
class DatasetKITTIOdo(BaseKittiDataset):
    def __init__(self, cfg: edict, engine_mode: EngineMode = EngineMode.TRAIN) -> None:
        super(DatasetKITTIOdo, self).__init__(cfg, engine_mode)
        self.train_dataset = []
        self.dataset_path = os.path.dirname(self._cfg['root_folder'])
        if engine_mode == EngineMode.TRAIN:
            assert 'train_sequences' in cfg, 'train_sequences must be provided in the configuration'
            for sequence in cfg['train_sequences']:
                self.process_sequence(sequence)
                self.train_dataset.append(pykitti.odometry(self.dataset_path, sequence))
                velo2cam2 = torch.from_numpy(self.train_dataset[-1].calib.T_cam2_velo).float()
                self.T_cam2_velo[sequence] = velo2cam2
                velo2cam3 = torch.from_numpy(self.train_dataset[-1].calib.T_cam3_velo).float()
                self.T_cam3_velo[sequence] = velo2cam3
        else:
            self.process_sequence(self._cfg['test_sequence'])
            self.test_dataset = pykitti.odometry(self.dataset_path, self._cfg['test_sequence'])
            velo2cam2 = torch.from_numpy(self.test_dataset.calib.T_cam2_velo).float()
            self.T_cam2_velo[self._cfg['test_sequence']] = velo2cam2
            velo2cam3 = torch.from_numpy(self.test_dataset.calib.T_cam3_velo).float()
            self.T_cam3_velo[self._cfg['test_sequence']] = velo2cam3
        
        self.test_RT = self.get_test_RT()
    
    
    def process_sequence(self, sequence: str) -> None:
        """
        Process KITTI Odometry sequence.
        
        Args:
            sequence: Sequence identifier.
        """
        image_files = sorted(os.listdir(os.path.join(self._cfg['root_folder'], sequence, 'image_2')))
        for image_file in image_files:
            timestamp_formatted = image_file.split('.')[0]
            map_file_path = [self._cfg['root_folder'], sequence, self._cfg['pcl_folder'], 
                           timestamp_formatted + '.bin']
            img_file_path = [self._cfg['root_folder'], sequence, self._cfg['imgs_folder'], 
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
    def get_camera_parameters(self, path: str) -> tuple[CameraIntrinsicParameters, torch.Tensor]:
        """
        Get camera intrinsic parameters.
        
        Args:
            path: Path to the calibration file.
            
        Returns:
            Tuple of (camera_intrinsic_parameters, camera_extrinsic_parameters).
        """
        calib_txt_path = os.path.join(self._cfg['root_folder'], path, 'calib.txt')
        if not is_path_exist(calib_txt_path):
            raise FileNotFoundError(f"File Not Found: {calib_txt_path}")
        calib = read_txt_file(calib_txt_path)
        
        if self._cfg['image'] == 'img2':
            camera_intrinsic_parameters = CameraIntrinsicParameters(
                calib['P2'][0], calib['P2'][5], calib['P2'][2], calib['P2'][6]
            )
        elif self._cfg['image'] == 'img3':
            camera_intrinsic_parameters = CameraIntrinsicParameters(
                calib['P3'][0], calib['P3'][5], calib['P3'][2], calib['P3'][6]
            )
        else:
            raise TypeError("image Not Available")
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
            self._cfg['root_folder'], sequence, 'velodyne', timestamp + '.bin'
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
            self._cfg['root_folder'], sequence, self._cfg['imgs_folder'],
            timestamp, 'image.jpg'
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

        item = self.all_files[idx]
        camera_parameters_path = str(item.split('/')[0])
        camera_intrinsic_parameters, camera_extrinsic_parameters = self.get_camera_parameters(
            camera_parameters_path
        )
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
