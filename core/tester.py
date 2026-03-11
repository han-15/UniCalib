import time
import csv
from matplotlib import cm
from .model import *
from .evaluation import *
from core.deepL.engine import SingleTester
from core.deepL.datasets import get_test_data_loader
from core.utils import ensure_dir
from core.deepL.model import create_model
from core.deepL.evaluation import get_evaluation
from core.deepL.datasets.data_preprocess import DepthFlowGenerator

from core.constant.deepL import EngineMode
from core.deepL.tensor_ops import flow2image,get_rotation_translation_from_transform


import torch

@torch.no_grad()

class Tester(SingleTester):
    def __init__(self):
        super().__init__()

        # dataloader
        start_time = time.time()
        data_loader = get_test_data_loader(self._cfg)
        loading_time = time.time() - start_time
        self.log(f"Data loader created: {loading_time:.3f}s collapsed.", level="DEBUG")
        self.register_loader(data_loader)
        # model
        model = create_model(self._cfg)
        self.register_model(model)
        # evaluator
        self.eval_func = get_evaluation('FlowEvalFunction', self._cfg)
        # preparation
        self.depth_flow_generator = DepthFlowGenerator(self._cfg)

    
    def overlay_imgs(self, rgb, lidar):
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]

        rgb = rgb.clone().cpu().permute(1,2,0).numpy()
        rgb = rgb*std+mean

        lidar[lidar == 0] = 1000.
        lidar = -lidar

        lidar = lidar.clone()
        lidar = lidar.unsqueeze(0)
        lidar = lidar.unsqueeze(0)

        lidar = F.max_pool2d(lidar, 3, 1, 1)
        lidar = -lidar
        lidar[lidar == 1000.] = 0.

        lidar = lidar[0][0]

        lidar = lidar.cpu().numpy()
        min_d = 0
        max_d = np.max(lidar)
        lidar = ((lidar - min_d) / (max_d - min_d)) * 255
        lidar = lidar.astype(np.uint8)
        lidar_color = cm.jet(lidar)
        lidar_color[:, :, 3] = 0.5
        lidar_color[lidar == 0] = [0, 0, 0, 0]
        blended_img = lidar_color[:, :, :3] * (np.expand_dims(lidar_color[:, :, 3], 2)) \
                    + rgb * (1. - np.expand_dims(lidar_color[:, :, 3], 2))
        blended_img = blended_img.clip(min=0., max=1.)

        blended_img = cv2.cvtColor((blended_img*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        # cv2.imwrite(f'./images/output/{idx:06d}_{name}.png', blended_img)
        return blended_img
    def overlay_lidar(self, lidar,if_estimated=False):
        # set white background
        background = np.ones((320, 960, 3)) * 1.0
    
        if if_estimated:
            lidar =(lidar+1)/2
            lidar=torch.exp(lidar*np.log(101))-1
            lidar=lidar/100.

        lidar_processed = lidar.clone()
        lidar_processed[lidar_processed == 0] = 1000.
        lidar_processed = -lidar_processed
    
        lidar_processed = lidar_processed.unsqueeze(0).unsqueeze(0)
        lidar_processed = F.max_pool2d(lidar_processed, 3, 1, 1)
        lidar_processed = -lidar_processed
        lidar_processed[lidar_processed == 1000.] = 0.
    
        lidar_processed = lidar_processed[0][0]
        lidar_np = lidar_processed.cpu().numpy()
        
        min_d=0
        max_d=np.max(lidar_np)
        lidar = ((lidar_np - min_d) / (max_d - min_d)) * 255
        lidar_uint8 = lidar.astype(np.uint8)
        # lidar_uint8 = (lidar_np * 255).astype(np.uint8)
        
        # apply color mapping
        lidar_color = cm.jet(lidar_uint8)
        lidar_color[:, :, 3] = 1.0  # set alpha
        
        # only valid pixels get color, invalid pixels stay transparent
        valid_mask = lidar_np > 0
        lidar_color[~valid_mask] = [0, 0, 0, 0]
        
        # blend image
        alpha_channel = np.expand_dims(lidar_color[:, :, 3], 2)
        blended_img = (lidar_color[:, :, :3] * alpha_channel + 
                       background * (1. - alpha_channel))
        blended_img = blended_img.clip(min=0., max=1.)
    
        # convert color space
        blended_img = cv2.cvtColor((blended_img*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return blended_img
    

    
    # save original results (after applying initial perturbation)
    def render(self, iteration, data_dict, output_dict):
        result_dir = self._cfg.experiment.result_dir / f"iter_{iteration}"
        ensure_dir(result_dir)
        # vision_image = data_dict['vision_images_input'][0]
        vision_image = data_dict['original_images_input'][0]
        estimated_vision_depth_img=data_dict['vision_images_input'][0]
        depth_image =data_dict['depth_images_input'][0]
        lidar_mask = data_dict['lidar_mask'][0]

        depth_image_gt = data_dict['depth_images_fine'][0]
        depth_image_gt_dense= data_dict['dense_gt_lidar_depth'][0]


        # 1. Original RGB image
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
        vision_image_np = vision_image.clone().cpu().permute(1,2,0).numpy()
        vision_image_np = vision_image_np*std+mean
        vision_image_np = cv2.cvtColor((vision_image_np*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{result_dir}/01_rgb_original.png', vision_image_np)

        # 2. Monocular depth estimation
        cv2.imwrite(f'{result_dir}/02_mono_depth_estimated.png', self.overlay_lidar(estimated_vision_depth_img[0,:,:], True))

        # 3. Ground truth
        cv2.imwrite(f'{result_dir}/03_gt_lidar_sparse.png', self.overlay_lidar(depth_image_gt[0,:,:]))
        cv2.imwrite(f'{result_dir}/04_gt_lidar_dense.png', self.overlay_lidar(depth_image_gt_dense[0,:,:]))
        cv2.imwrite(f'{result_dir}/05_gt_lidar_sparse_overlay.png', self.overlay_imgs(vision_image, depth_image_gt[0,:,:]))

        # 4. LiDAR depth after initial perturbation (mis-calibration)
        cv2.imwrite(f'{result_dir}/06_perturbed_lidar_dense.png', self.overlay_lidar(depth_image[0,:,:]))
        cv2.imwrite(f'{result_dir}/07_perturbed_lidar_sparse.png', self.overlay_lidar(lidar_mask[0,:,:]))
        cv2.imwrite(f'{result_dir}/08_perturbed_lidar_dense_overlay.png', self.overlay_imgs(vision_image, depth_image[0,:,:]))
        cv2.imwrite(f'{result_dir}/09_perturbed_lidar_sparse_overlay.png', self.overlay_imgs(vision_image, lidar_mask[0,:,:]))

    # save results after predicted extrinsic transformation
    def render_pre(self, iteration, data_dict, output_dict):
        vision_image = data_dict['original_images_input'][0]
        depth_image =  data_dict['depth_images_input'][0]
        lidar_mask = data_dict['lidar_mask'][0]
        flow_image_pred = output_dict['final'][0]

        result_dir = self._cfg.experiment.result_dir / f"iter_{iteration}"
        ensure_dir(result_dir)

        # dense lidar depth map after predicted transform
        cv2.imwrite(f'{result_dir}/10_dense_lidar_pre.png', self.overlay_imgs(vision_image, depth_image[0,:,:]))
        # sparse lidar depth map after predicted transform
        cv2.imwrite(f'{result_dir}/11_sparse_lidar_pre.png', self.overlay_imgs(vision_image, lidar_mask[0,:,:]))
        # predicted optical flow map
        cv2.imwrite(f'{result_dir}/12_flow_pred_transed.png', flow2image(flow_image_pred.permute(1, 2, 0).cpu().detach().numpy()))

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict, engine_mode=EngineMode.TEST)
        return output_dict

 
    def eval_step(self, iteration, data_dict, output_dict):

        result_dict = self.eval_func(data_dict, output_dict) 
        if self._cfg.experiment.if_render:
            self.render(iteration, data_dict, output_dict)
            r_error_inv = data_dict['rot_error'][0]
            t_error_inv = data_dict['tr_error'][0]
            rt_error_inv = get_transform_from_rotation_translation(r_error_inv, t_error_inv)
            pre_error = torch.linalg.inv(result_dict['predict'])
            rt = torch.matmul(rt_error_inv, pre_error)
            r,t = get_rotation_translation_from_transform(rt)
            data_dict['tr_error'] = [t]
            data_dict['rot_error'] = [r]
            data_dict = self.depth_flow_generator.push(data_dict, EngineMode.TEST)
            self.render_pre(iteration, data_dict, output_dict)
        result_dict['predict'] = 0
        return result_dict
    
    def before_test_step(self, iteration, data_dict):
        data_dict = self.depth_flow_generator.push(data_dict, EngineMode.TEST)
        return data_dict
    

    def after_test_epoch(self, result_dict):
        result_dir = self._cfg.experiment.result_dir
        ensure_dir(result_dir)
        result_file = result_dir / 'result_dict.csv'
        metrics=self._metrics_manager.get_metrics()
        with open(result_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write the header
            writer.writerow(metrics.keys())
            # Write the rows
            writer.writerows(zip(*metrics.values()))
    
