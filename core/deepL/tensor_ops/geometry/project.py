import torch
import numpy as np
from torch import Tensor
from ....constant import CameraIntrinsicParameters

def adjust_coordinate(xyz: Tensor, order=[1,2,0]) -> Tensor:
    """
    Adjusts the coordinates of a given tensor based on the specified order.

    Args:
        xyz (torch.Tensor): The input tensor containing coordinates. C * N.
        order (list, optional): A list specifying the new order of the coordinates. Defaults to [1, 2, 0, 3].

    Returns:
        torch.Tensor: The tensor with adjusted coordinates.
    """
    return xyz[order, :]

def project_with_mask(points: Tensor, image_size, camera_params: CameraIntrinsicParameters, adjust_coordinate_order=None, front=True) -> tuple[Tensor, Tensor, Tensor]:
    """
    Projects 3D points onto a 2D image plane using intrinsic camera parameters and returns the projected points,
    their depths, and a mask indicating valid points.
    Args:
        points (torch.Tensor): A 3xN tensor representing the 3D points to be projected.
        image_size (tuple): A tuple (width, height) representing the size of the image.
    Returns:
        tuple: A tuple containing:
            - uv (torch.Tensor): A 2xN_front tensor of the projected 2D points.
            - depth (torch.Tensor): A 1D tensor of the depths of the valid points.
            - mask (torch.Tensor): A 1D tensor indicating which points are valid after projection.
    Raises:
        TypeError: If the input points tensor does not have a shape of 3xN.
    """
    if points.shape[0] == 3:
        points = points
    elif points.shape[0] == 4:
        points = points[:3,:]
    else:
        raise TypeError("xyz must be a 3xN matrix. but xyz is ", points.shape)
    if adjust_coordinate_order is not None:
        points = adjust_coordinate(points, adjust_coordinate_order)  # (3, N)
    mask = torch.ones(points.shape[1], dtype=torch.bool, device=points.device) # (N)
    if front:
        mask_front = mask_pixels_with_front(points[2, :]) # (N)
        points = points[:, mask_front] # (3, N_front)
        mask = mask_front # (N)
    uv = torch.zeros((2, points.shape[1]), device=points.device) # (2, N_front)
    uv[0, :] = camera_params.focal_length_x * points[0, :] / points[2, :] + camera_params.principal_point_x
    uv[1, :] = camera_params.focal_length_y * points[1, :] / points[2, :] + camera_params.principal_point_y
    
    mask_vision = mask_pixels_with_vision(uv, (0.1, image_size[1]), (0.1, image_size[0])) # (N_front)
    # generate complete indexes
    index_front = torch.where(mask == True)[0] # (N_front)
    mask[index_front] = mask[index_front] & mask_vision # (N_front) in (N) & (N_front)

    return uv[:, mask_vision], points[2, mask_vision], mask

def mask_pixels_with_image_size(pixels: Tensor, image_w_range: tuple[float, float], image_h_range: tuple[float, float]) -> Tensor:
    """Compute the masks of the pixels which are within the range of an image.

    Args:
        pixels (Tensor): the pixels in the shape of (..., 2). Note that the pixels are represented as (h, w).
        image_w_range (tuple[float, float]): The range of the image width.
        image_h_range (tuple[float, float]): The range of the image height.

    Returns:
        A BoolTensor of the masks of the pixels in the shape of (..., 2). A pixel is with the image if True.
    """
    masks = torch.logical_and(
        torch.logical_and(torch.ge(pixels[0, ...], image_w_range[0]), torch.lt(pixels[0, ...], image_w_range[1])),
        torch.logical_and(torch.ge(pixels[1, ...], image_h_range[0]), torch.lt(pixels[1, ...], image_h_range[1])),
    )
    return masks

def mask_pixels_with_front(depth: Tensor) -> Tensor:
    """Compute the masks of the pixels which are in the front.

    Args:
        pixels (Tensor): the pixels in the shape of (..., 2). Note that the pixels are represented as (h, w).
        depth (Tensor): the depth tensor.

    Returns:
        A BoolTensor of the masks of the pixels in the shape of (..., 2). A pixel is in the front if True.
    """
    return torch.ge(depth, 0)

def mask_pixels_with_vision(pixels: Tensor, image_w_range: tuple[float, float], image_h_range: tuple[float, float], depth:Tensor=None, front=False):
    """
    Masks the pixels based on vision information.

    Args:
        pixels (Tensor): the pixels in the shape of (..., 2). Note that the pixels are represented as (h, w).
        image_w_range (tuple[float, float]): The range of the image width.
        image_h_range (tuple[float, float]): The range of the image height.
        depth (Tensor, optional): The depth tensor. Defaults to None.
        front (bool, optional): Whether to mask pixels in the front. Defaults to False.

    Returns:
        Tensor: The masked pixels tensor.
    """
    masks = mask_pixels_with_image_size(pixels, image_w_range, image_h_range)
    if front:
        assert depth is not None, "depth should be provided when front is True"
        masks = torch.logical_and(masks, mask_pixels_with_front(depth))
    return masks

def deproject(uv, pc_project_uv, camera_params: CameraIntrinsicParameters):
    index = np.argwhere(uv > 0)
    mask = uv > 0
    z = uv[mask]

    camera_params = camera_params.cpu().numpy()
    x = (index[:, 1] - camera_params[2]) * z / camera_params[0]
    y = (index[:, 0] - camera_params[3]) * z / camera_params[1]
    zxy = np.array([z, x, y])
    zxy = torch.tensor(zxy, dtype=torch.float32)
    zxyw = torch.cat([zxy, torch.ones(1, zxy.shape[1], device=zxy.device)])
    zxy = zxyw[:3, :]
    zxy = zxy.cpu().numpy()
    xyz = zxy[[1, 2, 0], :]

    # apply mask to pc_project_uv
    pc_project_u = pc_project_uv[:, :, 0][mask]
    pc_project_v = pc_project_uv[:, :, 1][mask]
    pc_project = np.array([pc_project_v, pc_project_u])
    match_index = np.array([index[:, 0], index[:, 1]])

    return xyz.transpose(), pc_project.transpose(), match_index.transpose()