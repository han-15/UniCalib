import numpy as np
import torch
import visibility
import cv2
from easydict import EasyDict as edict
from torch import Tensor
from core.deepL.tensor_ops import (
    TransformDistanceType,
    rotation_vector_to_rotation_matrix, rotation_matrix_to_quaternion, quaternion_to_rotation_matrix,
    transform_distance,
    get_transform_from_rotation_translation,
    inverse_rotation_translation,
    deproject
)
from core.deepL.evaluation import register_evaluation
from core.deepL.evaluation.loss import Evaluation
from core.constant.geometry import CameraIntrinsicParameters

MAX_FLOW = 400
EPSILON = 1e-9
FLOW_LOSS_THRESHOLD = 1e3
gamma = 0.8

@register_evaluation
class SequenceLossFunction(Evaluation):
    def __init__(self, cfg: edict):
        super().__init__(cfg)
        
    def evaluation_fn(self, data_dict, output_dict):
        """ Loss function defined over sequence of flow predictions """
        flow_gt = data_dict['flow_images_gt']
        valid = data_dict['valid_masks']
        
        mag = torch.sum(flow_gt**2, dim=1).sqrt() # (N, H, W)
        mask = (flow_gt[:, 0, :, :] != 0) | (flow_gt[:, 1, :, :] != 0)
        valid = (mag < MAX_FLOW) & mask
        
        n_predictions = len(output_dict['flow']) # number of iterations
        flow_loss = 0.0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            loss_i = output_dict['nf'][i]
            final_mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & valid[:, None]
            loss_reg = (final_mask * loss_i).sum() / (final_mask.sum() + EPSILON)
            flow_loss += i_weight * loss_reg.mean()
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            loss_i = output_dict['nf'][i]
            final_mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & valid[:, None]
            loss_reg = (final_mask * loss_i).sum() / (final_mask.sum() + EPSILON)
            flow_loss += i_weight * loss_reg.mean()
            
        epe = torch.sum((output_dict['final'] - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]
        
        if flow_loss > FLOW_LOSS_THRESHOLD or torch.isnan(flow_loss):
            flow_loss = torch.tensor(0.0, requires_grad=True)
        
        return {
                'loss': flow_loss, 
                'epe': epe.mean().item(),
                '1px': (epe < 1).float().mean().item(),
                '2px': (epe < 2).float().mean().item(),
                }

@register_evaluation
class SequenceEvalFunction(Evaluation):
    def __init__(self, cfg: edict):
        super().__init__(cfg)
        self.val_metric = 'val_epe'
    
    def evaluation_fn(self, data_dict, output_dict):
        flow_up = output_dict['final']
        flow_gt = data_dict['flow_images_gt']
        
        out_list, epe_list = [], []
        epe = torch.sum((flow_up - flow_gt) ** 2, dim=1).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        epe = epe.view(-1)
        mag = mag.view(-1)
        
        valid_gt = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)
        val = valid_gt.view(-1) >= 0.5
        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        epe_list = np.array(epe_list)
        out_list = np.concatenate(out_list)

        epe = np.mean(epe_list)
        f1 = 100 * np.mean(out_list)
        return {'val_epe': epe, 'val_f1': f1}

@register_evaluation
class FlowEvalFunction(Evaluation):
    def __init__(self, cfg: edict):
        super().__init__(cfg)
        self.count=0
    
    def flow_image2transform_with_depth_image(self, flow_image: Tensor, depth_image: Tensor, camera_params: CameraIntrinsicParameters):
        device = flow_image.device
        # create output tensor and pred_depth_img tensor
        output = torch.zeros(flow_image.shape).to(device)
        pred_depth_img = torch.zeros(depth_image.shape).to(device)
        pred_depth_img += 1000.
        # warp the depth image
        output: Tensor = visibility.image_warp_index(depth_image.to(device), flow_image.int(), pred_depth_img, output,
                                            depth_image.shape[3], depth_image.shape[2])
        pred_depth_img[pred_depth_img == 1000.] = 0.
        pc_project_uv = output.cpu().permute(0, 2, 3, 1).numpy()
        depth_img_ori = (torch.exp(depth_image.cpu() * torch.log(torch.tensor(101.))) - 1).numpy()

        # generate mask
        mask_depth_1 = pc_project_uv[0, :, :, 0] != 0
        mask_depth_2 = pc_project_uv[0, :, :, 1] != 0
        mask_depth = mask_depth_1 + mask_depth_2
        depth_img = depth_img_ori[0, 0, :, :] * mask_depth

        
        if 'Odo' in self._cfg.dataset.name: 
            h, w = 28, 140
        elif 'Raw' in self._cfg.dataset.name:  # adjust the principal point 1226*370 W * H
            h,w=25,133
        elif 'Waymo' in self._cfg.dataset.name: # adjust the principal point 1920*1280 W * H
            # h, w = 32, 0
            h, w = 160, 0
        elif 'Nuscenes' in self._cfg.dataset.name: 
            h, w = 20, 0
        elif '360' in self._cfg.dataset.name: # 1408 376
            h,w=28, 224
        elif 'CustomTest' in self._cfg.dataset.name:  #1920* 1080
            h, w = 100, 0
        camera_params.principal_point_x = camera_params.principal_point_x - w
        camera_params.principal_point_y = camera_params.principal_point_y - h
        camera_params_matrix = camera_params.to_matrix().numpy()
        # deproject the depth image
        pts3d, pts2d, _ = deproject(depth_img, pc_project_uv[0, :, :, :], camera_params)
        if pts3d.shape[0] < 4 or pts2d.shape[0] < 4:
            print('Not enough points for solvePnPRansac')
            rvecs = np.zeros((3, 1))
            tvecs = np.zeros((3, 1))
        _, rvecs, tvecs, _ = cv2.solvePnPRansac(pts3d, pts2d, camera_params_matrix, None)
        # convert the rotation vector to euler angles
        rotation_matrix = rotation_vector_to_rotation_matrix(rvecs)
        translation_vector = torch.tensor(tvecs).float()

        rotation_predicted, translation_predicted = inverse_rotation_translation(rotation_matrix, translation_vector)
        rotation_predicted_quaternion = rotation_matrix_to_quaternion(rotation_predicted)
        
        if 'KITTI' in self._cfg.dataset.name or 'Waymo' in self._cfg.dataset.name:
            rotation_predicted_quaternion = rotation_predicted_quaternion[0, [2, 0, 1, 3]]
        rotation_predicted = quaternion_to_rotation_matrix(rotation_predicted_quaternion)
        if 'KITTI' in self._cfg.dataset.name or 'Waymo' in self._cfg.dataset.name:
            translation_predicted = translation_predicted[:, [2, 0, 1]]
        
        transform_predicted = get_transform_from_rotation_translation(rotation_predicted, translation_predicted)
        
        return transform_predicted.to(device)
   
    def evaluation_fn(self, data_dict, output_dict):
        flow_image_predicted = output_dict['final']
        flow_gt = data_dict['flow_images_gt']
        depth_image = data_dict['lidar_mask']
        camera_params = data_dict['camera_intrinsic_parameters'][0]
        translation_error = data_dict['tr_error']
        rotation_error = data_dict['rot_error']

        transform_predicted= self.flow_image2transform_with_depth_image(flow_image_predicted, depth_image, camera_params.clone())
        
        rotation_distance, translation_distance = transform_distance(transform_predicted, 
                                                            get_transform_from_rotation_translation(rotation_error, translation_error), 
                                                            flag = TransformDistanceType.ALL)

        flow_up = output_dict['final']

        out_list, epe_list = [], []
        epe = torch.sum((flow_up - flow_gt) ** 2, dim=1).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        epe = epe.view(-1)
        mag = mag.view(-1)
        valid_gt = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)
        val = valid_gt.view(-1) >= 0.5
        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        epe_list = np.array(epe_list)
        out_list = np.concatenate(out_list)
        epe = np.mean(epe_list)
        f1 = 100 * np.mean(out_list)
        if isinstance(rotation_distance, list) and isinstance(translation_distance, list):
            return {
                't_Mean': translation_distance[0],
                'tX': translation_distance[1][0],
                'tY': translation_distance[1][1],
                'tZ': translation_distance[1][2],
                'R_Mean': rotation_distance[0],
                'RX': rotation_distance[1][0],
                'RY': rotation_distance[1][1],
                'RZ': rotation_distance[1][2],
                'EPE': epe,
                'F1': f1,
                'predict': transform_predicted
            }
        else:
            return {
                'Test_Trans_Error': translation_distance,
                'Test_Rotation_Error': rotation_distance,
                'Test_EPE': epe,
                'Test_F1': f1
            }
