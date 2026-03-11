from torch import Tensor
import torch
import numpy as np
from . import so3
from enum import Enum

def generate_random_translation(max_offset: float) -> list[float]:
    """
    Generate a random translation vector within the specified maximum offset.

    Args:
        max_offset (float): The maximum offset for each translation component.

    Returns:
        list[float]: A list containing the randomly generated translation vector [transl_x, transl_y, transl_z].
    """
    transl_x = np.random.uniform(-max_offset, max_offset)
    transl_y = np.random.uniform(-max_offset, max_offset)
    transl_z = np.random.uniform(-max_offset, min(max_offset, 1.))  
    return [transl_x, transl_y, transl_z]

def generate_random_transforms(max_angle: float, max_offset: float) -> Tensor:
    """Generate random rotation and translation.
    Args:
        batch_size (int): number of samples
        device (torch.device): device
    Returns:
        rotation (Tensor): (*, 3, 3)
        translation (Tensor): (*, 3)
    """
    rotation_euler = so3.generate_random_rotation_euler(max_angle)
    rotation_matrix_tensor = so3.angle_to_rotation_matrix(torch.tensor(rotation_euler))
    translation_tensor = torch.tensor(generate_random_translation(max_offset))
    return get_transform_from_rotation_translation(rotation_matrix_tensor, translation_tensor)

def apply_transform_to_points(points: Tensor, transform: Tensor) -> Tensor:
    """Rigid transform to points and normals (optional).

    Given a point cloud P(3, N), normals V(3, N) and a transform matrix T in the form of
      | R t |
      | 0 1 |,
    the output point cloud Q = RP + t, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T + t, V' = VR^T.

    There are three cases supported:
    1. points and normals are (*, 3), transform is (4, 4), the output points are (*, 3).
       In this case, the transform is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 4, 4), the output points are (B, N, 3).
       In this case, the transform is applied batch-wise. The points are automatically broadcast if B=1.
    3. points and normals are (B, 3), transform is (B, 4, 4), the output points are (B, 3).
       In this case, the points are automatically broadcast to (B, 1, 3) and the transform is applied batch-wise. The
       first dim of points/normals and transform must be the same.

    Args:
        points (Tensor): (*, 3) or (B, N, 3) or (B, 3).
        normals (optional[Tensor]=None): same shape as points.
        transform (Tensor): (4, 4) or (B, 4, 4)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    assert transform.dim() == 2 or (
        transform.dim() == 3 and points.dim() in [2, 3]
    ), f"Incompatible shapes between points {tuple(points.shape)} and transform {tuple(transform.shape)}."

    if transform.dim() == 2:
        # case 1: (*, 3) x (4, 4)
        input_shape = points.shape
        rotation = transform[:3, :3]  # (3, 3)
        translation = transform[None, :3, 3]  # (1, 3)
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.reshape(*input_shape)
    elif transform.dim() == 3 and points.dim() == 3:
        # case 2: (B, N, 3) x (B, 4, 4)
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
    elif transform.dim() == 3 and points.dim() == 2:
        # case 3: (B, 3) x (B, 4, 4)
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = points.unsqueeze(1)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.squeeze(1)
    
    return points

def get_rotation_translation_from_transform(transform: Tensor) -> tuple[Tensor, Tensor]:
    """Decompose transformation matrix into rotation matrix and translation vector.
    Args:
        transform (Tensor): (*, 4, 4)
    Returns:
        rotation (Tensor): (*, 3, 3)
        translation (Tensor): (*, 3)
    """
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    return rotation, translation

def get_transform_from_rotation_translation(rotation: Tensor, translation: Tensor) -> Tensor:
    """Compose transformation matrix from rotation matrix and translation vector.
    Args:
        rotation (Tensor): (*, 3, 3) 
        translation (Tensor): (*, 3) 
    Returns:
        transform (Tensor): (*, 4, 4) float
    """
    input_shape = rotation.shape
    rotation = rotation.view(-1, 3, 3)
    translation = translation.view(-1, 3)
    transform = torch.eye(4).to(rotation).unsqueeze(0).repeat(rotation.shape[0], 1, 1)
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = translation
    output_shape = input_shape[:-2] + (4, 4)
    transform = transform.view(*output_shape)
    return transform.float()

def get_quaternion_from_transform(transform: Tensor) -> Tensor:
    """Convert transformation matrix to quaternion.
    Args:
        transform (Tensor): (*, 4, 4)
    Returns:
        quaternion (Tensor): (*, 4)
    """
    rotation, _ = get_rotation_translation_from_transform(transform)  # (*, 3, 3)
    quaternion = so3.rotation_matrix_to_quaternion(rotation)  # (*, 4)
    return quaternion

def inverse_transform(transform: Tensor) -> Tensor:
    """Inverse rigid transform.
    Args:
        transform (Tensor): (*, 4, 4) 
    Return:
        inv_transform (Tensor): (*, 4, 4)
    """
    rotation, translation = get_rotation_translation_from_transform(transform)  # (*, 3, 3), (*, 3)
    inv_rotation = rotation.transpose(-1, -2)  # (*, 3, 3)
    inv_translation = -torch.matmul(inv_rotation, translation.unsqueeze(-1)).squeeze(-1)  # (*, 3)
    inv_transform = get_transform_from_rotation_translation(inv_rotation, inv_translation)  # (*, 4, 4)
    return inv_transform

def inverse_rotation_translation(rotation: Tensor, translation: Tensor) -> tuple[Tensor, Tensor]:
    """
    Inverse rotation and translation.
    
    Args:
        rotation (Tensor): Rotation matrix with shape (*, 3, 3).
        translation (Tensor): Translation vector with shape (*, 3). Note: actual shape is (3, 1).
        
    Returns:
        Tuple of (inv_rotation, inv_translation):
            - inv_rotation (Tensor): Inverse rotation matrix with shape (*, 3, 3).
            - inv_translation (Tensor): Inverse translation vector with shape (*, 3).
    """
    inv_rotation = rotation.transpose(-1, -2).float()  # (*, 3, 3)
    inv_translation = -torch.matmul(inv_rotation, translation.view(-1,3,1).float()).squeeze(-1)  # (*, 3)
    return inv_rotation, inv_translation

class TransformDistanceType(Enum):
    """Distance type for rigid transformations."""
    COMMON = 0
    I2D_LOC = 1
    MEAN = 2
    ALL = 3

def transform_distance(transform1: Tensor, transform2: Tensor, flag=TransformDistanceType.I2D_LOC) -> tuple[Tensor, Tensor]:
    """Compute distance between two rigid transformations.
    Args:
        transform1 (Tensor): (*, 4, 4)
        transform2 (Tensor): (*, 4, 4)
    Returns:
        distance(Tensor, Tensor): rotation distance, translation distance
    """
    match flag:
        case TransformDistanceType.COMMON:
            rotation1, translation1 = get_rotation_translation_from_transform(transform1)  # (*, 3, 3), (*, 3)
            rotation2, translation2 = get_rotation_translation_from_transform(transform2)  # (*, 3, 3), (*, 3)
            rotation_distance = so3.rotation_matrix_distance(rotation1, rotation2)  # (*,)
            translation_distance = torch.norm(translation1 - translation2, dim=-1)  # (*,)
        case TransformDistanceType.I2D_LOC:
            rotation, translation = get_rotation_translation_from_transform(transform = torch.matmul(inverse_transform(transform2), transform1))
            rotation_distance = so3.quaternion_distance(
                so3.rotation_matrix_to_quaternion(rotation), 
                torch.tensor([[0., 0., 0., 1.]])
            ) * 180. / torch.pi
            translation_distance = torch.norm(translation) * 100  # (*,)
        case TransformDistanceType.MEAN:
            # Calculate relative transformation
            rotation, translation = get_rotation_translation_from_transform(
                transform = torch.matmul(inverse_transform(transform2), transform1)
            )
            
            # Calculate rotation error (in degrees)
            rotation_errors = so3.rotation_matrix_to_angle(rotation)    
            rotation_distance = torch.mean(rotation_errors.abs())
            
            # Calculate translation error (in cm)
            # translation shape: [1, 1, 3]
            translation = translation.squeeze(0)  # [1, 3]
            translation_errors = translation.abs() * 100  # Convert to cm
            translation_distance = torch.mean(translation_errors)
        case TransformDistanceType.ALL:
            # Calculate relative transformation
            rotation, translation = get_rotation_translation_from_transform(
                transform = torch.matmul(inverse_transform(transform2), transform1)
            )
            
            # Calculate rotation error (in degrees)
            rotation_errors = so3.rotation_matrix_to_angle(rotation)
            rotation_mean_error = torch.mean(rotation_errors.abs())
            rotation_distance = [rotation_mean_error.item()] + rotation_errors.abs().tolist()
            
            # Calculate translation error (in cm)
            translation = translation.squeeze(0)  # [1, 3]
            translation_errors = translation.abs() * 100  # Convert to cm
            translation_mean_error = torch.mean(translation_errors)
            translation_distance = [translation_mean_error.item()] + translation_errors.tolist()
            
    return rotation_distance, translation_distance