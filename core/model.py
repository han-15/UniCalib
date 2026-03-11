import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from .layers import (
    conv3x3,
    coords_grid,
    CorrBlock,
    InputPadder,
    BasicUpdateBlock,
    ResNetFPN,
    ResNetFPNMask
)
from core.deepL.model import register_model
from core.constant.deepL import EngineMode

@register_model
class RAFT(nn.Module):
    def __init__(self, cfg: edict):
        super(RAFT, self).__init__()
        self.cfg = cfg
        self.output_dim = cfg.model.dim * 2

        # Set the number of levels and radius for correlation pyramid
        self.cfg.model.corr_levels = 4
        self.cfg.model.corr_radius = cfg.model.radius
        self.cfg.model.corr_channel = cfg.model.corr_levels * (cfg.model.radius * 2 + 1) ** 2
        
        # Initialize context network
        self.cnet = ResNetFPN(cfg, input_dim=2, output_dim=self.cfg.model.dim * 2, norm_layer=nn.BatchNorm2d, init_weight=True)

        # Initialize convolution layers
        self.init_conv = conv3x3(2 * cfg.model.dim, 2 * cfg.model.dim)
        # Initialize upsampling weights
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(cfg.model.dim, cfg.model.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.model.dim * 2, 64 * 9, 1, padding=0)
        )
        # Initialize flow head
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(cfg.model.dim, 2 * cfg.model.dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * cfg.model.dim, 6, 3, padding=1)
        )

        # If number of iterations > 0, initialize feature network and update block
        if cfg.model.iters > 0:
            self.fnet_lidar = ResNetFPNMask(cfg, input_dim=1, output_dim=self.output_dim, norm_layer=nn.InstanceNorm2d, init_weight=True)
            self.update_block = BasicUpdateBlock(cfg, hdim=cfg.model.dim, cdim=cfg.model.dim)

    def upsample_data(self, flow, info, mask):
        """ Upsample [H/8, W/8, C] -> [H, W, C] using convex combination """
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)
        
        return up_flow.reshape(N, 2, 8*H, 8*W), up_info.reshape(N, C, 8*H, 8*W)

    def forward(self, data_dict, engine_mode=EngineMode.TRAIN):
        """ Estimate optical flow between pair of frames """
        image1 = data_dict['depth_images_input']
        image2 = data_dict['vision_images_input']
        lidar_mask = data_dict['lidar_mask']
        
        N, _, H, W = image1.shape
        iters = self.cfg.model.iters
        image1 = 2 * image1 - 1.0  # normalize to [-1, 1]    
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        flow_predictions = []
        info_predictions = []

        # Pad images
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        N, _, H, W = image1.shape
        dilation = torch.ones(N, 1, H//8, W//8, device=image1.device)
        
        # run the context network
        cnet = self.cnet(torch.cat([image1, image2], dim=1))
        cnet = self.init_conv(cnet)
        net, context = torch.split(cnet, [self.cfg.model.dim, self.cfg.model.dim], dim=1)

        # init flow
        flow_update = self.flow_head(net)
        weight_update = .25 * self.upsample_weight(net)
        flow_8x = flow_update[:, :2]
        info_8x = flow_update[:, 2:]
        flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
        flow_predictions.append(flow_up)
        info_predictions.append(info_up)
            
        if self.cfg.model.iters > 0:
            # run the feature network
            fmap1_8x = self.fnet_lidar(image1,lidar_mask)
            fmap2_8x = self.fnet_lidar(image2)
            corr_fn = CorrBlock(fmap1_8x, fmap2_8x, self.cfg)

        for itr in range(iters):
            N, _, H, W = flow_8x.shape
            flow_8x = flow_8x.detach()
            coords2 = (coords_grid(N, H, W, device=image1.device) + flow_8x).detach()
            corr = corr_fn(coords2, dilation=dilation)
            net = self.update_block(net, context, corr, flow_8x)
            flow_update = self.flow_head(net)
            weight_update = .25 * self.upsample_weight(net)
            flow_8x = flow_8x + flow_update[:, :2]
            info_8x = flow_update[:, 2:]
            # upsample predictions
            flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        for i in range(len(flow_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])

        if engine_mode == EngineMode.TRAIN or engine_mode == EngineMode.TEST:
            flow_gt = data_dict['flow_images_gt'] if data_dict['flow_images_gt'] is not None else torch.zeros(N, 2, H, W, device=image1.device)
            
            nf_predictions = []
            for i in range(len(info_predictions)):
                if not self.cfg.model.use_var:
                    var_max = var_min = 0
                else:
                    var_max = self.cfg.model.var_max
                    var_min = self.cfg.model.var_min
                    
                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]  #α             
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)  #var_min=0
                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                nf_predictions.append(nf_loss)

            return {'final': flow_predictions[-1], 'flow': flow_predictions, 'info': info_predictions, 'nf': nf_predictions}
        else:
            return {'final': flow_predictions[-1], 'flow': flow_predictions, 'info': None, 'nf': None}


