import abc
import numpy as np
import torch

from .base_dataset import BaseDataset
from ...constant import EngineMode


class BaseKittiDataset(BaseDataset, abc.ABC):
    """
    Base KITTI dataset class that extracts common methods for KITTI series datasets.
    
    Args:
        cfg: Configuration object containing dataset parameters.
        engine_mode: Engine mode (TRAIN, VALID, or TEST).
    """
    
    def __init__(self, cfg, engine_mode: EngineMode = EngineMode.TRAIN):
        super(BaseKittiDataset, self).__init__(cfg, engine_mode)
        self.T_cam2_velo = {}
        self.T_cam3_velo = {}

    def load_velo_scan(self, file):
        """
        Load and parse velodyne binary file.
        
        Args:
            file: Path to the velodyne binary file.
            
        Returns:
            Point cloud array with shape (N, 4).
        """
        scan = np.fromfile(file, dtype=np.float32)
        return scan.reshape((-1, 4))

    def adjust_kitti_point_cloud(self, pc: np.ndarray, sequence) -> torch.Tensor:
        """
        Preprocess KITTI point cloud and transform to camera coordinate system.
        
        Args:
            pc: Point cloud array.
            sequence: Sequence identifier.
            
        Returns:
            Point cloud in camera coordinates with shape (4, N).
        """
        pc_in = self.adjust_point_cloud(pc)
        
        if self._cfg['image'] == 'img2':
            pc_in = torch.mm(self.T_cam2_velo[sequence], pc_in)
        elif self._cfg['image'] == 'img3':
            pc_in = torch.mm(self.T_cam3_velo[sequence], pc_in)
        else:
            raise TypeError("image Not Available")
        
        pc_in = pc_in[[2, 0, 1, 3], :]
        return pc_in

    @abc.abstractmethod
    def get_image_path(self, idx) -> str:
        """
        Get image file path. Must be implemented by subclasses.
        
        Args:
            idx: Index of the data sample.
            
        Returns:
            Path to the image file.
        """
        pass
