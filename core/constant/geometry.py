import torch
from torch import Tensor

class CameraIntrinsicParameters(Tensor):
    def __new__(cls, focal_length_x, focal_length_y, principal_point_x, principal_point_y):
        data = torch.tensor([focal_length_x, focal_length_y, principal_point_x, principal_point_y], dtype=torch.float32)
        return Tensor._make_subclass(cls, data, data.requires_grad)

    def __init__(self, focal_length_x, focal_length_y, principal_point_x, principal_point_y):
        pass

    @property
    def focal_length_x(self):
        return self[0]

    @focal_length_x.setter
    def focal_length_x(self, value):
        self[0] = value

    @property
    def focal_length_y(self):
        return self[1]

    @focal_length_y.setter
    def focal_length_y(self, value):
        self[1] = value

    @property
    def principal_point_x(self):
        return self[2]

    @principal_point_x.setter
    def principal_point_x(self, value):
        self[2] = value

    @property
    def principal_point_y(self):
        return self[3]

    @principal_point_y.setter
    def principal_point_y(self, value):
        self[3] = value

    def to_matrix(self) -> Tensor:
        return torch.tensor([[self.focal_length_x, 0, self.principal_point_x],
                             [0, self.focal_length_y, self.principal_point_y],
                             [0, 0, 1.]], dtype=torch.float32)

    def scale(self, w_scale: float, h_scale: float) -> 'CameraIntrinsicParameters':
        self.focal_length_x *= w_scale
        self.focal_length_y *= h_scale
        self.principal_point_x *= w_scale
        self.principal_point_y *= h_scale
        return self
