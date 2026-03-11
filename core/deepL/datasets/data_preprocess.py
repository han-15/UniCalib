import torch
import visibility
import numpy as np
import cv2
from easydict import EasyDict as edict
from ...deepL.tensor_ops import (
    get_transform_from_rotation_translation, 
    apply_transform_to_points, 
    get_flow_image_from_flow_set, 
    get_flow_set_from_2pixel_sets,
    project_with_mask,
)
from ...constant import CameraIntrinsicParameters, EngineMode

def logarithmic_normalize(depth_image, max_depth=100.):
    """
    Apply logarithmic normalization to depth image, range 0-1.
    
    Args:
        depth_image: Input depth image tensor.
        max_depth: Maximum depth value for clamping.
        
    Returns:
        Normalized depth image.
    """
    depth_image = torch.clamp(depth_image, min=0, max=max_depth)
    normalized = torch.log(depth_image + 1) / torch.log(torch.tensor(max_depth + 1))
    return normalized

def sparse_to_dense(sparse, max_depth=100.):
    """
    Convert sparse depth map to dense depth map using morphological operations.
    
    Args:
        sparse: Sparse depth map.
        max_depth: Maximum depth value.
        
    Returns:
        Dense depth map.
    """
    valid = sparse > 0.1
    sparse[valid] = max_depth - sparse[valid]

    dilate_kernel = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)
    sparse = cv2.dilate(sparse, dilate_kernel)

    close_kernel = np.ones((5, 5), np.uint8)
    sparse = cv2.morphologyEx(sparse, cv2.MORPH_CLOSE, close_kernel)

    invalid = sparse < 0.1
    fill_kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(sparse, fill_kernel)
    sparse[invalid] = dilated[invalid]

    valid = sparse > 0.1
    sparse[valid] = max_depth - sparse[valid]

    return sparse

def dilation(lidar_input):
    depth_img_input = []
    for i in range(lidar_input.shape[0]):
        depth_img = lidar_input[i, 0, :, :].cpu().numpy() * 100.
        depth_img_dilate = sparse_to_dense(depth_img.astype(np.float32)) / 100.
        depth_img_input.append(depth_img_dilate)
    
    depth_img_input = torch.tensor(np.array(depth_img_input)).float().to(lidar_input.device)
    depth_img_input = depth_img_input.unsqueeze(1)
    return depth_img_input

class DepthFlowGenerator:
    def __init__(self, cfg: edict):
        self._real_shape = None
        self._cfg = cfg
        self._occlusion_threshold = cfg.dataset.occlusion_threshold
        self._occlusion_kernel = cfg.dataset.occlusion_kernel

    def gen_depth_img(self, uv, depth, index, cam_params: CameraIntrinsicParameters):
        """
        Generate depth image and occlusion mask.
        
        Args:
            uv: Pixel coordinates.
            depth: Depth values.
            index: Point indices.
            cam_params: Camera intrinsic parameters.
            
        Returns:
            Tuple of (depth_image, deocclusion_mask, mask_image, mask_image).
        """
        device = uv.device

        depth_image = torch.zeros(self._real_shape[:2], device=device, dtype=torch.float) + 1000.
        mask_image = (-1) * torch.ones(self._real_shape[:2], device=device, dtype=torch.float)
        index = index.float()
        
        depth_image, mask_image = visibility.depth_image(
            uv, depth, index, depth_image, mask_image, uv.shape[0],
            self._real_shape[1], self._real_shape[0]
        )
        depth_image[depth_image == 1000.] = 0.
        
        mask_image_deocclusion = (-1) * torch.ones(self._real_shape[:2], device=device, dtype=torch.float)
        depth_image_no_occlusion = torch.zeros_like(depth_image, device=device)

        depth_image_no_occlusion, mask_image_deocclusion = visibility.visibility2(
            depth_image, cam_params.to(device), mask_image,
            depth_image_no_occlusion, mask_image_deocclusion,
            depth_image.shape[1], depth_image.shape[0],
            self._occlusion_threshold, self._occlusion_kernel
        )
        return depth_image_no_occlusion, mask_image_deocclusion.int(), mask_image, mask_image

    def flatten_mask(self, mask_deocclusion, range_mask_uv):
        index_deocclusion = torch.where(mask_deocclusion > 0)
        mask_deocclusion = mask_deocclusion[index_deocclusion[0][:], index_deocclusion[1][:]]
        mask_flatten = torch.zeros(range_mask_uv.shape[0], device=mask_deocclusion.device, dtype=torch.int32)
        mask_flatten[mask_deocclusion.cpu().numpy() - 1] = mask_deocclusion
        return mask_flatten

    def crop_data_from_dict(self, data: dict, patch_shape, engine_mode: EngineMode):
        H, W = patch_shape[:2]
        patch_H, patch_W = patch_shape[2:]
        assert patch_H <= H and patch_W <= W, f"Patch size ({patch_H}, {patch_W}) should be smaller than the image size ({H}, {W})"
        if engine_mode == EngineMode.TRAIN:
            x = np.random.randint(0, H - patch_H) if H > patch_H else 0
            y = np.random.randint(0, W - patch_W) if W > patch_W else 0
        else:
            x = (H - patch_H) // 2
            y = (W - patch_W) // 2
        # Unpack Data from Dict
        return {key: value[..., x:x + patch_H, y:y + patch_W] for key, value in data.items()}
    
    def crop_data_from_dict_with_intrinsic(self, data: dict, cam_params: CameraIntrinsicParameters, 
                                           patch_shape, engine_mode: EngineMode):
        """
        Crop image data and adjust camera intrinsic parameters accordingly.
        
        Args:
            data: Dictionary containing image data.
            cam_params: Camera intrinsic parameters.
            patch_shape: Shape tuple (H, W, patch_H, patch_W).
            engine_mode: Engine mode (TRAIN, VALID, or TEST).
            
        Returns:
            Tuple of (cropped_data_dict, adjusted_camera_parameters).
        """
        H, W = patch_shape[:2]
        patch_H, patch_W = patch_shape[2:]
        assert patch_H <= H and patch_W <= W, \
            f"Patch size ({patch_H}, {patch_W}) should be smaller than the image size ({H}, {W})"
        
        if engine_mode == EngineMode.TRAIN:
            x = np.random.randint(0, H - patch_H) if H > patch_H else 0
            y = np.random.randint(0, W - patch_W) if W > patch_W else 0
        else:
            x = (H - patch_H) // 2
            y = (W - patch_W) // 2
        
        cropped_data = {key: value[..., x:x + patch_H, y:y + patch_W] for key, value in data.items()}
        adjusted_cam_params = CameraIntrinsicParameters(
            cam_params.focal_length_x,
            cam_params.focal_length_y,
            cam_params.principal_point_x - y,
            cam_params.principal_point_y - x
        )
        
        return cropped_data, adjusted_cam_params

    def push(self, data_dict: dict, engine_mode=EngineMode.TRAIN):
        """
        Process data dictionary to generate depth images, optical flow, etc.
        
        Args:
            data_dict: Dictionary containing input data.
            engine_mode: Engine mode (TRAIN, VALID, or TEST).
            
        Returns:
            Updated data dictionary with processed data.
        """
        vision_images = data_dict['vision_image']
        point_clouds = data_dict['point_cloud']
        camera_intrinsic_parameters = data_dict['camera_intrinsic_parameters']
        T_errs = data_dict['tr_error']
        R_errs = data_dict['rot_error']
        orders = data_dict['order']
        device = vision_images[0].device
        
        vision_images_input = []
        depth_images_input = []
        depth_images_fine = []
        flow_images_gt = []
        valid_masks = []
        adjusted_camera_intrinsic_parameters = []
        
        if engine_mode == EngineMode.TEST:
            original_images = data_dict['original_image']
            original_images_input = []
            
        for idx in range(len(vision_images)):
            # 1 Unpack Data
            vision_image = vision_images[idx].to(device)
            if engine_mode == EngineMode.TEST:
                original_image = original_images[idx].to(device)
            point_cloud_fine = point_clouds[idx].clone().to(device)
            cam_params = CameraIntrinsicParameters(
                camera_intrinsic_parameters[idx][0],
                camera_intrinsic_parameters[idx][1],
                camera_intrinsic_parameters[idx][2],
                camera_intrinsic_parameters[idx][3]
            )
            order = orders[idx]
            self._real_shape = [
                int(vision_image.shape[1]), 
                int(vision_image.shape[2]), 
                vision_image.shape[0]
            ]
            
            # Transform Point Cloud
            transform_fine2coarse = get_transform_from_rotation_translation(
                R_errs[idx].to(device), T_errs[idx].to(device)
            ).squeeze(0)
            point_cloud_coarse = apply_transform_to_points(
                point_cloud_fine[:3].transpose(-1, -2), transform_fine2coarse
            ).transpose(-1, -2)
            
            # Project Point Cloud
            uv_fine, depth_fine, mask_fine = project_with_mask(
                point_cloud_fine, self._real_shape, cam_params, order
            )
            uv_fine = uv_fine.t().int().contiguous()
            uv_coarse, depth_coarse, mask_coarse = project_with_mask(
                point_cloud_coarse, self._real_shape, cam_params, order
            )
            uv_coarse = uv_coarse.t().int().contiguous()
            
            # Get Flow Set
            flow_set, mask_flow = get_flow_set_from_2pixel_sets(
                uv_coarse, uv_fine, mask_coarse, mask_fine
            )
            
            # Filter flow points in coarse points
            mask_flow_coarse = mask_coarse[mask_flow]
            range_mask_uv_coarse = torch.arange(mask_flow_coarse.shape[0]).to(device) + 1
            uv_coarse_in_flow = uv_coarse[mask_flow[mask_coarse], :]
            depth_coarse_in_flow = depth_coarse[mask_flow[mask_coarse]]
            
            # Filter flow points in fine points
            mask_flow_fine = mask_fine[mask_flow]
            range_mask_uv_fine = torch.arange(mask_flow_fine.shape[0]).to(device) + 1
            uv_fine_in_flow = uv_fine[mask_flow[mask_fine], :]
            depth_fine_in_flow = depth_fine[mask_flow[mask_fine]]
            
            # Get Deocclusion Mask
            _, mask_deocclusion_coarse, _, _ = self.gen_depth_img(
                uv_coarse_in_flow, depth_coarse_in_flow, range_mask_uv_coarse, cam_params
            )
            mask_depth_coarse = self.flatten_mask(mask_deocclusion_coarse, range_mask_uv_coarse)
            _, mask_deocclusion_fine, _, _ = self.gen_depth_img(
                uv_fine_in_flow, depth_fine_in_flow, range_mask_uv_fine, cam_params
            )
            mask_depth_fine = self.flatten_mask(mask_deocclusion_fine, range_mask_uv_fine)

            # Get Depth Image for Training
            depth_image, _, _, _ = self.gen_depth_img(
                uv_coarse, depth_coarse, mask_coarse[mask_coarse], cam_params
            )
            depth_image = logarithmic_normalize(depth_image, max_depth=100.)

            depth_image = depth_image.unsqueeze(0)

            mask_depth = (mask_depth_coarse > 0) & (mask_depth_fine > 0)
            
            depth_image_fine, _, _, _ = self.gen_depth_img(
                uv_fine, depth_fine, mask_fine[mask_fine], cam_params
            )
            depth_image_fine = logarithmic_normalize(depth_image_fine, max_depth=100.)

            depth_image_fine = depth_image_fine.unsqueeze(0)
            
            # Get Flow Image
            flow_image = get_flow_image_from_flow_set(
                flow_set, uv_coarse_in_flow, mask_depth, self._real_shape[:2]
            )
            
            # Crop Data with Intrinsic Adjustment
            if engine_mode != EngineMode.TEST:
                cropped_data, adjusted_cam_params = self.crop_data_from_dict_with_intrinsic(
                    dict(image=vision_image, depth=depth_image, flow=flow_image),
                    cam_params,
                    vision_image.shape[-2:] + (320, 960),
                    engine_mode
                )
                vision_image, depth_image, flow_image = cropped_data.values()
            else:
                cropped_data, adjusted_cam_params = self.crop_data_from_dict_with_intrinsic(
                    dict(original_image=original_image, image=vision_image, depth=depth_image,
                         depth_fine=depth_image_fine, flow=flow_image),
                    cam_params,
                    vision_image.shape[-2:] + (320, 960),
                    engine_mode
                )
                original_image, vision_image, depth_image, depth_image_fine, flow_image = cropped_data.values()
            
            if len(adjusted_camera_intrinsic_parameters) == 0:
                adjusted_camera_intrinsic_parameters.append(adjusted_cam_params)
            
            valid_i = (flow_image[0].abs() < 1000) & (flow_image[1].abs() < 1000)

            vision_images_input.append(vision_image)
            depth_images_input.append(depth_image)
            flow_images_gt.append(flow_image)
            valid_masks.append(valid_i)
            if engine_mode == EngineMode.TEST:
                original_images_input.append(original_image)
                depth_images_fine.append(depth_image_fine)

            dense_lidar_depth_input = dilation(torch.stack(depth_images_input))
            if engine_mode == EngineMode.TEST:
                dense_gt_lidar_depth = dilation(torch.stack(depth_images_fine))
                
        if engine_mode != EngineMode.TEST:
            data_dict.update({
                "vision_images_input": torch.stack(vision_images_input),
                "depth_images_input": dense_lidar_depth_input,
                "lidar_mask": torch.stack(depth_images_input),
                "flow_images_gt": torch.stack(flow_images_gt),
                "valid_masks": torch.stack(valid_masks),
            })
        else:
            data_dict.update({
                "vision_images_input": torch.stack(vision_images_input),
                "depth_images_input": dense_lidar_depth_input,
                "lidar_mask": torch.stack(depth_images_input),
                "flow_images_gt": torch.stack(flow_images_gt),
                "valid_masks": torch.stack(valid_masks),
                "original_images_input": torch.stack(original_images_input),
                "depth_images_fine": torch.stack(depth_images_fine),
                "dense_gt_lidar_depth": dense_gt_lidar_depth,
            })
        return data_dict