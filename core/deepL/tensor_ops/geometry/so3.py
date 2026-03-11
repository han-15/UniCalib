from scipy.spatial.transform import Rotation as R
from torch import Tensor
import torch
import numpy as np

def generate_random_rotation_euler(max_angle: float) -> list[float]:
    """
    Generates a random rotation in Euler angles representation.

    Args:
        max_angle (float): The maximum angle in radians for each Euler angle.

    Returns:
        Tensor: A tensor representing the rotation in Euler angles.

    """
    rotation_euler = [np.random.uniform(-max_angle, max_angle) for _ in range(3)]
    return rotation_euler

def angle_to_rotation_matrix(angle: Tensor, degrees: bool=True) -> Tensor:
    """Convert angle to rotation matrix. 
    Args:
        angle (Tensor): (*, 3)
        degrees (bool): True means degrees, False means radians.
    Returns:
        rotation_matrix (Tensor): (*, 3, 3) float
    """
    angle = angle.view(-1, 3)
    return torch.tensor(np.array([R.from_euler('xyz', angle[i].detach().cpu().numpy(), degrees=degrees).as_matrix() 
                         for i in range(angle.shape[0])])).float()

def rotation_matrix_distance(rotation_matrix1: Tensor, rotation_matrix2: Tensor) -> Tensor:
    """Compute the distance between two rotation matrices. The error unit of the calculation is the rotation angle error.
    Args:
        rotation_matrix1 (Tensor): (*, 3, 3)
        rotation_matrix2 (Tensor): (*, 3, 3)
    Returns:
        distance (Tensor): (*)
    """
    rotation_matrix1 = rotation_matrix1.view(-1, 3, 3)
    rotation_matrix2 = rotation_matrix2.view(-1, 3, 3)
    return torch.tensor([abs(torch.acos((torch.trace(torch.mm(torch.inverse(rotation_matrix1).view(3,3),rotation_matrix2.view(3,3)))-1)/2)) * 180. / np.pi
                        for i in range(rotation_matrix1.shape[0])])

# Quaternion
def quaternion_to_rotation_matrix(quaternion: Tensor) -> Tensor:
    """Convert quaternion to rotation matrix.
    Args:
        quaternion (Tensor): (*, 4)
    Returns:
        rotation_matrix (Tensor): (*, 3, 3) float
    """
    quaternion = quaternion.view(-1, 4)
    return torch.tensor(np.array([R.from_quat(quaternion[i].detach().cpu().numpy()).as_matrix() 
                         for i in range(quaternion.shape[0])])).float()

def rotation_matrix_to_quaternion(rotation_matrix: Tensor) -> Tensor:
    """
    Convert rotation matrix to quaternion.
    
    Args:
        rotation_matrix (Tensor): Rotation matrix with shape (*, 3, 3).
        
    Returns:
        quaternion (Tensor): Quaternion with shape (*, 4) in [xyzw] format.
    """
    rotation_matrix = rotation_matrix.view(-1, 3, 3)
    return torch.tensor(np.array([R.from_matrix(rotation_matrix[i].detach().cpu().numpy()).as_quat() 
                         for i in range(rotation_matrix.shape[0])]))

def quaternion_inverse(quaternion: Tensor) -> Tensor:
    """Compute the inverse of a quaternion.
    Args:
        quaternion (Tensor): (*, 4)
    Returns:
        quaternion_inv (Tensor): (*, 4)
    """
    quaternion = quaternion.view(-1, 4)
    return torch.tensor(np.array([R.from_quat(quaternion[i].detach().cpu().numpy()).inv().as_quat() 
                         for i in range(quaternion.shape[0])]))

def quaternion_multiply(quaternion1: Tensor, quaternion2: Tensor) -> Tensor:
    """
    Compute the multiplication of two quaternions.
    
    Args:
        quaternion1 (Tensor): First quaternion with shape (*, 4) in [xyzw] format.
        quaternion2 (Tensor): Second quaternion with shape (*, 4) in [xyzw] format.
        
    Returns:
        quaternion (Tensor): Result quaternion with shape (*, 4) in [xyzw] format.
    """
    q = quaternion1.view(-1, 4)[:,[3,0,1,2]]
    r = quaternion2.view(-1, 4)[:,[3,0,1,2]]
    t = torch.zeros(q.shape[0], 4, device=q.device)
    t[:, 0] = r[:, 0] * q[:, 0] - r[:, 1] * q[:, 1] - r[:, 2] * q[:, 2] - r[:, 3] * q[:, 3]
    t[:, 1] = r[:, 0] * q[:, 1] + r[:, 1] * q[:, 0] - r[:, 2] * q[:, 3] + r[:, 3] * q[:, 2]
    t[:, 2] = r[:, 0] * q[:, 2] + r[:, 1] * q[:, 3] + r[:, 2] * q[:, 0] - r[:, 3] * q[:, 1]
    t[:, 3] = r[:, 0] * q[:, 3] - r[:, 1] * q[:, 2] + r[:, 2] * q[:, 1] + r[:, 3] * q[:, 0]
    return t

def quaternion_distance(quaternion1: Tensor, quaternion2: Tensor) -> Tensor:
    """
    Compute the distance between two quaternions. The error unit of the calculation is the rotation angle error.
    
    Args:
        quaternion1 (Tensor): First quaternion with shape (*, 4).
        quaternion2 (Tensor): Second quaternion with shape (*, 4).
        
    Returns:
        distance (Tensor): Distance in radians with shape (*).
    """
    t = quaternion_multiply(quaternion1, quaternion_inverse(quaternion2))
    return 2 * torch.atan2(torch.norm(t[:, 1:], dim=1), torch.abs(t[:, 0]))

# Rotation Vector
def rotation_vector_to_rotation_matrix(rotation_vector: Tensor) -> Tensor:
    """Convert rotation vector to rotation matrix.
    Args:
        rotation_vector (Tensor): (*, 3)
    Returns:
        rotation_matrix (Tensor): (*, 3, 3)
    """
    rotation_vector = torch.tensor(rotation_vector).view(-1, 3)
    return torch.tensor(np.array([R.from_rotvec(rotation_vector[i].detach().cpu().numpy()).as_matrix() 
                         for i in range(rotation_vector.shape[0])]))

def rotation_matrix_to_angle(rotation: Tensor, degrees: bool=True) -> Tensor:
    """Convert rotation matrix to angle.
    Args:
        rotation (Tensor): (*, 3, 3)
        degrees (bool): True means degrees, False means radians.
    Returns:
        angle (Tensor): (*, 3)
    """
    rotation = rotation.view(-1, 3, 3)

    angles = np.array([R.from_matrix(rotation[i].detach().cpu().numpy()).as_euler('xyz', degrees=degrees) 
                       for i in range(rotation.shape[0])])
    return torch.tensor(angles)