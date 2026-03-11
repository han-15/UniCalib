import torch
import numpy as np
from enum import Enum
from torch import Tensor

class SPLIT_TYPE(Enum):
    """
    Enum class representing the different types of data splits.

    Attributes:
        TRAIN (int): Represents the training data split.
        VALID (int): Represents the validation data split.
        TEST (int): Represents the test data split.
    """
    TRAIN = 0
    VALID = 1
    TEST = 2

class CAMERA_PARAMS:
    def __init__(self, focal_length: Tensor = torch.tensor(np.asarray([0, 0])), principal_point: Tensor = torch.tensor(np.asarray([0, 0]))):
        self.FOCAL_LENGTH = focal_length
        self.PRINCIPAL_POINT = principal_point

    def to(self, device) -> 'CAMERA_PARAMS':
        if self.is_numpy():
            self.tensor()
        self.FOCAL_LENGTH = self.FOCAL_LENGTH.to(device)
        self.PRINCIPAL_POINT = self.PRINCIPAL_POINT.to(device)
        return self
    
    def numpy(self) -> None:
        if self.is_tensor():
            self.FOCAL_LENGTH = self.FOCAL_LENGTH.numpy()
            self.PRINCIPAL_POINT = self.PRINCIPAL_POINT.numpy()
        
        assert self.is_numpy(), "The camera parameters are not in numpy format."

    def tensor(self) -> None:
        if self.is_numpy():
            self.FOCAL_LENGTH = torch.tensor(self.FOCAL_LENGTH)
            self.PRINCIPAL_POINT = torch.tensor(self.PRINCIPAL_POINT)
        assert self.is_tensor(), "The camera parameters are not in tensor format."
    
    def is_tensor(self) -> bool:
        return isinstance(self.FOCAL_LENGTH, Tensor) and isinstance(self.PRINCIPAL_POINT, Tensor)
    
    def is_numpy(self) -> bool:
        return isinstance(self.FOCAL_LENGTH, np.ndarray) and isinstance(self.PRINCIPAL_POINT, np.ndarray)
    
    def to_matrix(self) -> Tensor:
        return torch.tensor([[self.FOCAL_LENGTH[0], 0, self.PRINCIPAL_POINT[0]],
                             [0, self.FOCAL_LENGTH[1], self.PRINCIPAL_POINT[1]],
                             [0, 0, 1.]])
    
    