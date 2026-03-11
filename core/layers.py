import torch
import torch.nn as nn
import torch.nn.functional as F

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    
# Calculate correlation between feature maps
class CorrBlock:
    def __init__(self, fmap1, fmap2, cfg):
        self.num_levels = cfg.model.corr_levels
        self.radius = cfg.model.corr_radius
        self.cfg = cfg
        self.corr_pyramid = []
        # all pairs correlation
        for i in range(self.num_levels):
            corr = CorrBlock.corr(fmap1, fmap2, 1)
            batch, h1, w1, dim, h2, w2 = corr.shape
            corr = corr.reshape(batch*h1*w1, dim, h2, w2)
            fmap2 = F.interpolate(fmap2, scale_factor=0.5, mode='bilinear', align_corners=False)
            self.corr_pyramid.append(corr)

    def __call__(self, coords, dilation=None):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        if dilation is None:
            dilation = torch.ones(batch, 1, h1, w1, device=coords.device)

        # print(dilation.max(), dilation.mean(), dilation.min())
        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            device = coords.device
            dx = torch.linspace(-r, r, 2*r+1, device=device)
            dy = torch.linspace(-r, r, 2*r+1, device=device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            delta_lvl = delta_lvl * dilation.view(batch * h1 * w1, 1, 1, 1)
            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            coords_lvl = centroid_lvl + delta_lvl
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        out = out.permute(0, 3, 1, 2).contiguous().float()  
        return out

    @staticmethod
    def corr(fmap1, fmap2, num_head):
        batch, dim, h1, w1 = fmap1.shape
        h2, w2 = fmap2.shape[2:]
        fmap1 = fmap1.view(batch, num_head, dim // num_head, h1*w1)
        fmap2 = fmap2.view(batch, num_head, dim // num_head, h2*w2) 
        corr = fmap1.transpose(2, 3) @ fmap2
        corr = corr.reshape(batch, num_head, h1, w1, h2, w2).permute(0, 2, 3, 1, 4, 5)
        return corr  / torch.sqrt(torch.tensor(dim).float())

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, output_dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * output_dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * output_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.final = nn.Conv2d(dim, output_dim, kernel_size=1, padding=0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.final(input + x)
        return x

class BasicMotionEncoder(nn.Module):
    def __init__(self, cfg, dim=128):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = cfg.model.corr_channel
        self.convc1 = nn.Conv2d(cor_planes, dim*2, 1, padding=0)
        self.convc2 = nn.Conv2d(dim*2, dim+dim//2, 3, padding=1)
        self.convf1 = nn.Conv2d(2, dim, 7, padding=3)
        self.convf2 = nn.Conv2d(dim, dim//2, 3, padding=1)
        self.conv = nn.Conv2d(dim*2, dim-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicUpdateBlock(nn.Module):
    def __init__(self, cfg, hdim=128, cdim=128):
        #net: hdim, inp: cdim
        super(BasicUpdateBlock, self).__init__()
        self.cfg = cfg.model
        self.encoder = BasicMotionEncoder(cfg, dim=cdim)
        self.refine = []
        for i in range(cfg.model.num_blocks):
            self.refine.append(ConvNextBlock(2*cdim+hdim, hdim))
        self.refine = nn.ModuleList(self.refine)

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        for blk in self.refine:
            net = blk(torch.cat([net, inp], dim=1))
        return net

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()

        # self.sparse = sparse
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = norm_layer(planes)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.bn3 = norm_layer(planes)
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                self.bn3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.relu(self.bn2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x+y)

class ResNetFPN(nn.Module):
    """
    ResNet18, output resolution is 1/8.
    Each block has 2 layers.
    """
    def __init__(self, cfg, input_dim=3, output_dim=256, ratio=1.0, norm_layer=nn.BatchNorm2d, init_weight=False):
        super().__init__()
        # Config
        block = BasicBlock
        block_dims = cfg.model.block_dims
        initial_dim = cfg.model.initial_dim
        self.init_weight = init_weight
        self.input_dim = input_dim
        # Class Variable
        self.in_planes = initial_dim
        for i in range(len(block_dims)):
            block_dims[i] = int(block_dims[i] * ratio)
        # Networks
        self.conv1 = nn.Conv2d(input_dim, initial_dim, kernel_size=7, stride=2, padding=3)
        self.bn1 = norm_layer(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        if cfg.model.pretrain == 'resnet34':
            n_block = [3, 4, 6]
        elif cfg.model.pretrain == 'resnet18':
            n_block = [2, 2, 2]
        else:
            raise NotImplementedError       
        self.layer1 = self._make_layer(block, block_dims[0], stride=1, norm_layer=norm_layer, num=n_block[0])  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2, norm_layer=norm_layer, num=n_block[1])  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2, norm_layer=norm_layer, num=n_block[2])  # 1/8
        self.final_conv = conv1x1(block_dims[2], output_dim)
        self._init_weights(cfg)

    def _init_weights(self, cfg):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.init_weight:
            from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
            #modified
            if cfg.model.pretrain == 'resnet18':
                pretrained_dict = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
            else:
                pretrained_dict = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            #modified
            if self.input_dim == 6:
                for k, v in pretrained_dict.items():
                    if k == 'conv1.weight':
                        pretrained_dict[k] = torch.cat((v, v), dim=1)
            if self.input_dim == 1:
                for k, v in pretrained_dict.items():
                    if k == 'conv1.weight':
                        # Average weights from 3 channels to 1 channel
                        pretrained_dict[k] = v.mean(dim=1, keepdim=True)
            if self.input_dim == 4:
                for k, v in pretrained_dict.items():
                    if k == 'conv1.weight':
                        # Average weights from 3 channels to 1 channel, then concatenate
                        pretrained_dict[k] = torch.cat((v, v.mean(dim=1, keepdim=True)), dim=1)
            if self.input_dim == 2:
                for k, v in pretrained_dict.items():
                    if k == 'conv1.weight':
                        # Both channels are averaged from original 3 channels to 1 channel
                        pretrained_dict[k] = torch.cat((v.mean(dim=1, keepdim=True), v.mean(dim=1, keepdim=True)), dim=1)

                        
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
        
    def _make_layer(self, block, dim, stride=1, norm_layer=nn.BatchNorm2d, num=2):
        layers = []
        layers.append(block(self.in_planes, dim, stride=stride, norm_layer=norm_layer))
        for i in range(num - 1):
            layers.append(block(dim, dim, stride=1, norm_layer=norm_layer))
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x = self.relu(self.bn1(self.conv1(x)))
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
        # Output
        output = self.final_conv(x)
        return output

class ResNetFPNMask(nn.Module):
    """
    ResNet18, output resolution is 1/8.
    Each block has 2 layers.
    """
    def __init__(self, cfg, input_dim=3, output_dim=256, ratio=1.0, norm_layer=nn.BatchNorm2d, init_weight=False):
        super().__init__()
        # Config
        block = BasicBlock
        block_dims = cfg.model.block_dims
        initial_dim = cfg.model.initial_dim
        self.init_weight = init_weight
        self.input_dim = input_dim
        # Class Variable
        self.in_planes = initial_dim
        for i in range(len(block_dims)):
            block_dims[i] = int(block_dims[i] * ratio)
        # Networks
        self.conv1 = nn.Conv2d(input_dim, initial_dim, kernel_size=7, stride=2, padding=3)
        self.bn1 = norm_layer(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        self.mask_conv = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3)
        if cfg.model.pretrain == 'resnet34':
            n_block = [3, 4, 6]
        elif cfg.model.pretrain == 'resnet18':
            n_block = [2, 2, 2]
        else:
            raise NotImplementedError       
        self.layer1 = self._make_layer(block, block_dims[0], stride=1, norm_layer=norm_layer, num=n_block[0])  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2, norm_layer=norm_layer, num=n_block[1])  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2, norm_layer=norm_layer, num=n_block[2])  # 1/8
        self.final_conv = conv1x1(block_dims[2], output_dim)
        self._init_weights(cfg)

    def _init_weights(self, cfg):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.init_weight:
            from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
            if cfg.model.pretrain == 'resnet18':
                pretrained_dict = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
            else:
                pretrained_dict = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if self.input_dim == 6:
                for k, v in pretrained_dict.items():
                    if k == 'conv1.weight':
                        pretrained_dict[k] = torch.cat((v, v), dim=1)
            if self.input_dim == 1:
                for k, v in pretrained_dict.items():
                    if k == 'conv1.weight':
                        # Average weights from 3 channels to 1 channel
                        pretrained_dict[k] = v.mean(dim=1, keepdim=True)
            if self.input_dim == 4:
                for k, v in pretrained_dict.items():
                    if k == 'conv1.weight':
                        # Average weights from 3 channels to 1 channel, then concatenate
                        pretrained_dict[k] = torch.cat((v, v.mean(dim=1, keepdim=True)), dim=1)
            if self.input_dim == 2:
                for k, v in pretrained_dict.items():
                    if k == 'conv1.weight':
                        # Average weights from 3 channels to 1 channel, then duplicate to two channels
                        pretrained_dict[k] = torch.cat((v.mean(dim=1, keepdim=True), v.mean(dim=1, keepdim=True)), dim=1)

            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
        
    def _make_layer(self, block, dim, stride=1, norm_layer=nn.BatchNorm2d, num=2):
        layers = []
        layers.append(block(self.in_planes, dim, stride=stride, norm_layer=norm_layer))
        for i in range(num - 1):
            layers.append(block(dim, dim, stride=1, norm_layer=norm_layer))
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        # ResNet Backbone
        x = self.relu(self.bn1(self.conv1(x)))
        if mask is not None:
            mask = self.relu(self.bn1(self.mask_conv(mask)))
            x = x * mask
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
        # Output
        output = self.final_conv(x)
        return output
    