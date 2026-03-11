import abc
import numpy as np
import torch
import torchvision.transforms.functional as TTF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ...constant import CameraIntrinsicParameters, EngineMode
from ...utils import read_csv_file, is_path_exist, create_df, write_csv_file, get_logger
from ..tensor_ops import (
    angle_to_rotation_matrix, generate_random_transforms,
    get_rotation_translation_from_transform, inverse_rotation_translation,
    get_transform_from_rotation_translation, apply_transform_to_points,
    inverse_transform
)


class BaseDataset(Dataset, abc.ABC):
    """
    Base dataset class that extracts common methods for all datasets.
    
    Args:
        cfg: Configuration object containing dataset parameters.
        engine_mode: Engine mode (TRAIN, VALID, or TEST).
    """
    
    def __init__(self, cfg, engine_mode: EngineMode = EngineMode.TRAIN):
        super(BaseDataset, self).__init__()
        self._cfg = cfg
        self._engine_mode = engine_mode
        self._w_scale = cfg.w_scale
        self._h_scale = cfg.h_scale
        self._adjust_coordinate_order = cfg.adjust_coordinate_order if cfg.adjust_coordinate_order != "" else None
        self.all_files = []
        self.test_RT = []

    @abc.abstractmethod
    def process_sequence(self, sequence: str) -> None:
        """
        Process a sequence. Must be implemented by subclasses.
        
        Args:
            sequence: Sequence identifier.
        """
        pass

    @abc.abstractmethod
    def get_camera_parameters(self, *args, **kwargs):
        """
        Get camera parameters. Must be implemented by subclasses.
        
        Returns:
            Camera intrinsic and extrinsic parameters.
        """
        pass

    @abc.abstractmethod
    def get_point_cloud_path(self, idx):
        """
        Get point cloud file path. Must be implemented by subclasses.
        
        Args:
            idx: Index of the data sample.
            
        Returns:
            Path to the point cloud file.
        """
        pass

    @abc.abstractmethod
    def get_depth_image_path(self, idx):
        """
        Get depth image file path. Must be implemented by subclasses.
        
        Args:
            idx: Index of the data sample.
            
        Returns:
            Path to the depth image file.
        """
        pass

    def get_test_RT(self) -> list:
        """
        Generate or load random transformations for the test set.
        
        Returns:
            List of random rotation and translation parameters.
        """
        test_RT = []
        if self._engine_mode == EngineMode.TRAIN:
            return test_RT
        
        test_RT_file = self._get_test_RT_filename()
        if not is_path_exist(test_RT_file):
            get_logger().success(f'TEST SET - Not found: {test_RT_file}, Generating a new one')
            rad_factor = np.pi / 180.0
            len_files = len(self.all_files)
            data = {
                'tx': np.random.uniform(-self._cfg['max_t'], self._cfg['max_t'], len_files),
                'ty': np.random.uniform(-self._cfg['max_t'], self._cfg['max_t'], len_files),
                'tz': np.random.uniform(-self._cfg['max_t'], min(self._cfg['max_t'], 1.0), len_files),
                'rx': np.random.uniform(-self._cfg['max_r'], self._cfg['max_r'], len_files) * rad_factor,
                'ry': np.random.uniform(-self._cfg['max_r'], self._cfg['max_r'], len_files) * rad_factor,
                'rz': np.random.uniform(-self._cfg['max_r'], self._cfg['max_r'], len_files) * rad_factor
            }
            write_csv_file(create_df(data), test_RT_file)
        
        get_logger().success(f'TEST SET: Using this file: {test_RT_file}')
        test_RT.extend(read_csv_file(test_RT_file, sep=',').values.tolist())
        # assert len(test_RT) == len(self.all_files), f"Something wrong {len(test_RT)} != {len(self.all_files)}"
        return test_RT

    @abc.abstractmethod
    def _get_test_RT_filename(self) -> str:
        """
        Get test RT filename. Must be implemented by subclasses.
        
        Returns:
            Path to the test RT file.
        """
        pass

    def adjust_point_cloud(self, pc: np.ndarray) -> torch.Tensor:
        """
        Convert point cloud to homogeneous coordinate system (4, N).
        
        Args:
            pc: Point cloud array.
            
        Returns:
            Point cloud in homogeneous coordinates with shape (4, N).
        """
        if isinstance(pc, torch.Tensor):
            pc_in = pc
        else:
            pc_in = torch.from_numpy(pc.astype(np.float32))
        
        if pc_in.shape[1] in [3, 4]:
            pc_in = pc_in.t()
        
        if pc_in.shape[0] == 3:
            homogeneous = torch.ones(pc_in.shape[1]).unsqueeze(0)
            pc_in = torch.cat((pc_in, homogeneous), 0)
        elif pc_in.shape[0] == 4:
            if not torch.all(pc_in[3, :] == 1.):
                pc_in[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")
        
        return pc_in

    def transform_depth_tensor(self, depth: torch.Tensor, flip: bool = False, rotation_angle: float = 0.0) -> torch.Tensor:
        """
        Apply flip and rotation transformations to depth tensor.
        
        Args:
            depth: Depth tensor.
            flip: Whether to apply horizontal flip.
            rotation_angle: Rotation angle in degrees.
            
        Returns:
            Transformed depth tensor.
        """
        if not isinstance(depth, torch.Tensor):
            depth = torch.tensor(depth)
        
        if depth.dim() == 3:
            depth = depth.unsqueeze(0)
        
        if flip:
            depth = TTF.hflip(depth)
        
        if rotation_angle != 0:
            depth = TTF.rotate(depth, angle=rotation_angle)
        
        return depth.squeeze(0)

    def custom_transform_depth(self, depth, img_rotation=0., flip=False):
        """
        Data augmentation for depth images.
        
        Args:
            depth: Depth image.
            img_rotation: Rotation angle in degrees.
            flip: Whether to apply horizontal flip.
            
        Returns:
            Augmented depth image.
        """
        if self._engine_mode == EngineMode.TRAIN:
            depth = self.transform_depth_tensor(depth, flip=flip, rotation_angle=img_rotation)
        else:
            if not isinstance(depth, torch.Tensor):
                depth = torch.tensor(depth)
        return depth

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        """
        Data augmentation for RGB images.
        
        Args:
            rgb: RGB image.
            img_rotation: Rotation angle in degrees.
            flip: Whether to apply horizontal flip.
            
        Returns:
            Augmented and normalized RGB image tensor.
        """
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if self._engine_mode == EngineMode.TRAIN:
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def generate_random_transforms(self, idx: int = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate random transformations.
        
        Args:
            idx: Index of the data sample (used in test mode).
            
        Returns:
            Tuple of (rotation_matrix, translation_vector).
        """
        if self._engine_mode == EngineMode.TRAIN:
            R, T = get_rotation_translation_from_transform(
                generate_random_transforms(self._cfg['max_r'], self._cfg['max_t'])
            )
        else:
            R = angle_to_rotation_matrix(torch.tensor(self.test_RT[idx][4:]), False)
            T = torch.tensor(self.test_RT[idx][1:4])
        return inverse_rotation_translation(R, T)

    def generate_zero_transforms(self, idx: int = None) -> tuple[torch.Tensor, torch.Tensor]:
        # keep device/precision consistent with the project
        dtype = torch.float32
        device = getattr(self, "_device", None)

        # identity rotation, zero translation (column vector shape 3x1, compatible with inverse_* implementation)
        R = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)  # (1, 3, 3)
        T = torch.zeros(1, 3, dtype=dtype, device=device)  # (1, 3)

        # explicitly via inverse_rotation_translation (inverse of I,0 is still I,0)
        return inverse_rotation_translation(R, T)  

        
    def scale_image(self, image, cam_params: CameraIntrinsicParameters):
        """
        Scale image and camera parameters.
        
        Args:
            image: Input image tensor.
            cam_params: Camera intrinsic parameters.
            
        Returns:
            Tuple of (scaled_image, scaled_camera_parameters).
        """
        if self._w_scale == 1 and self._h_scale == 1:
            return image, cam_params
        
        self._real_shape = [
            int(image.shape[1] * self._h_scale),
            int(image.shape[2] * self._w_scale),
            image.shape[0]
        ]
        downsample = transforms.Resize(self._real_shape[:2], interpolation=Image.NEAREST)
        image = downsample(image)
        cam_params.scale(self._w_scale, self._h_scale)
        return image, cam_params

    def scale_image_all(self, image, original_img, cam_params: CameraIntrinsicParameters):
        """
        Scale depth image, original image, and camera parameters.
        
        Args:
            image: Depth image tensor.
            original_img: Original RGB image tensor.
            cam_params: Camera intrinsic parameters.
            
        Returns:
            Tuple of (scaled_depth_image, scaled_original_image, scaled_camera_parameters).
        """
        if self._w_scale == 1 and self._h_scale == 1:
            return image, original_img, cam_params
        
        self._real_shape = [
            int(image.shape[1] * self._h_scale),
            int(image.shape[2] * self._w_scale),
            image.shape[0]
        ]
        downsample = transforms.Resize(self._real_shape[:2], interpolation=Image.NEAREST)
        image = downsample(image)
        original_img = downsample(original_img)
        cam_params.scale(self._w_scale, self._h_scale)
        return image, original_img, cam_params

    def augment_data(self, depth_image, point_cloud,
                    camera_intrinsic_parameters: CameraIntrinsicParameters,
                    camera_extrinsic_parameters: torch.Tensor = None) -> tuple:
        """
        Data augmentation for depth images and point clouds.
        
        Args:
            depth_image: Depth image.
            point_cloud: Point cloud.
            camera_intrinsic_parameters: Camera intrinsic parameters.
            camera_extrinsic_parameters: Camera extrinsic parameters (optional).
            
        Returns:
            Tuple of (augmented_depth_image, augmented_point_cloud, adjusted_camera_parameters).
        """
        if self._engine_mode != EngineMode.TRAIN:
            image = self.custom_transform_depth(depth_image)
            image, camera_intrinsic_parameters = self.scale_image(image, camera_intrinsic_parameters)
            if camera_extrinsic_parameters is not None:
                point_cloud = self.adjust_point_cloud(point_cloud)
                point_cloud = apply_transform_to_points(
                    point_cloud[:3].transpose(-1, -2), camera_extrinsic_parameters
                )
            return image, self.adjust_point_cloud(point_cloud), camera_intrinsic_parameters

        image_rotation = np.random.uniform(-5, 5)
        h_mirror = np.random.rand() > 0.5

        image = self.custom_transform_depth(depth_image, image_rotation, h_mirror)
        if h_mirror:
            point_cloud[1, :] *= -1
            camera_intrinsic_parameters.principal_point_x = image.shape[2] - camera_intrinsic_parameters.principal_point_x

        R = angle_to_rotation_matrix(torch.tensor([image_rotation, 0, 0]))
        T = torch.tensor([0., 0., 0.]).float()
        transform = inverse_transform(get_transform_from_rotation_translation(R, T))
        point_cloud = apply_transform_to_points(point_cloud[:3].transpose(-1, -2), transform)
        
        if camera_extrinsic_parameters is not None:
            point_cloud = self.adjust_point_cloud(point_cloud)
            point_cloud = apply_transform_to_points(
                point_cloud[:3].transpose(-1, -2), camera_extrinsic_parameters
            )
        point_cloud = self.adjust_point_cloud(point_cloud)

        image, camera_intrinsic_parameters = self.scale_image(image, camera_intrinsic_parameters)
        return image, point_cloud, camera_intrinsic_parameters

    def augment_data_all(self, depth_image, original_image, point_cloud,
                        camera_intrinsic_parameters: CameraIntrinsicParameters,
                        camera_extrinsic_parameters: torch.Tensor = None) -> tuple:
        """
        Data augmentation including original RGB image.
        
        Args:
            depth_image: Depth image.
            original_image: Original RGB image.
            point_cloud: Point cloud.
            camera_intrinsic_parameters: Camera intrinsic parameters.
            camera_extrinsic_parameters: Camera extrinsic parameters (optional).
            
        Returns:
            Tuple of (augmented_depth_image, augmented_original_image, augmented_point_cloud, adjusted_camera_parameters).
        """
        if original_image is None:
            raise ValueError("Original image is None")
        
        image = self.custom_transform_depth(depth_image)
        original_image = self.custom_transform(original_image)
        image, original_image, camera_intrinsic_parameters = self.scale_image_all(
            image, original_image, camera_intrinsic_parameters
        )
        
        if camera_extrinsic_parameters is not None:
            point_cloud = self.adjust_point_cloud(point_cloud)
            point_cloud = apply_transform_to_points(
                point_cloud[:3].transpose(-1, -2), camera_extrinsic_parameters
            )
        
        return image, original_image, self.adjust_point_cloud(point_cloud), camera_intrinsic_parameters

    def __len__(self) -> int:
        return len(self.all_files)
