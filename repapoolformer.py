# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
import torch.autograd.profiler as profiler

from timm.models._registry import register_model
from timm.layers import DropPath, trunc_normal_, PatchEmbed
from timm.layers.helpers import to_2tuple

from functools import partial
from einops import rearrange, reduce
import math
import copy

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            dim_in,
            dim_hidden=None,
            dim_out=None,
            bias=True,
            drop_path=0.,
            use_conv=False,
            channel_idle=False,
            act_layer=nn.GELU,
            feature_norm="LayerNorm",
            std=1.0,
            layer_scale=False,
            init_values=1e-5,
            idle_ratio=0.75):
            
        super().__init__()
        
        ######################## ↓↓↓↓↓↓ ########################
        # Hyperparameters
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden or dim_in
        self.dim_out = dim_out or dim_in
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        # Self-attention projections
        self.fc1 = nn.Conv2d(self.dim_in, self.dim_hidden, 1)
        self.fc2 = nn.Conv2d(self.dim_hidden, self.dim_out, 1)
        self.act = act_layer()
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        # Channel-idle
        self.channel_idle = channel_idle
        self.act_channels = int(dim_hidden * (1-idle_ratio))
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        self.feature_norm = feature_norm
        if self.feature_norm == "LayerNorm":
            self.norm1 = nn.GroupNorm(1, self.dim_in)
        elif self.feature_norm == "BatchNorm":
            self.norm1 = nn.BatchNorm2d(self.dim_in)
            self.norm2 = nn.BatchNorm2d(self.dim_hidden)
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None
        ######################## ↑↑↑↑↑↑ ########################
            
        ######################## ↓↓↓↓↓↓ ########################
        # Layer Scale
        self.layer_scale = layer_scale
        if self.layer_scale:
            self.ls = nn.Parameter(torch.ones((self.dim_out)) * init_values)
        ######################## ↑↑↑↑↑↑ ########################
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x, epoch: int= 0):
        B, C, H, W = x.shape
        ######################## ↓↓↓ 2-layer MLP ↓↓↓ ########################
        shortcut = x # B, N, C
        
        # 1st Feature normalization
        x = self.norm1(x)
        
        # FFN in
        x = self.fc1(x) # B, N, 4C
        
        # Activation
        if self.channel_idle:
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask[:,:self.act_channels, :, :] = True
            x = torch.where(mask, self.act(x), x)
        else:
            x = self.act(x)
        
        # 2nd Feature normalization
        if self.feature_norm == "BatchNorm":
            x = self.norm2(x)
            
        # FFN out
        x = self.fc2(x)
        
        # Add Layer Scale (dim)
        if self.layer_scale:
            x = x * self.ls.unsqueeze(-1).unsqueeze(-1)
        
        # Add DropPath
        x = self.drop_path(x) if self.drop_path is not None else x
        
        x = x + shortcut
        ######################## ↑↑↑ 2-layer MLP ↑↑↑ ########################
        #if x.get_device() == 0:
            #print("x after ffn:", x.std(-1).mean().item(), x.mean().item(), x.max().item(), x.min().item())
        return x
        
    def reparam(self):
        with torch.no_grad():
            mean = self.norm1.running_mean
            std = torch.sqrt(self.norm1.running_var + self.norm1.eps)
            weight = self.norm1.weight
            bias = self.norm1.bias
        
            fc1_bias = self.fc1((-mean/std*weight+bias)[None, :, None, None]).squeeze()
            fc1_weight = (self.fc1.weight / std[None, :, None, None] * weight[None, :, None, None]).squeeze()
                    
            mean = self.norm2.running_mean
            std = torch.sqrt(self.norm2.running_var + self.norm2.eps)
            weight = self.norm2.weight
            bias = self.norm2.bias
                    
            fc2_bias = self.fc2((-mean/std*weight+bias)[None, :, None, None]).squeeze()
            fc2_weight = (self.fc2.weight / std[None, :, None, None] * weight[None, :, None, None]).squeeze()
            
            if self.layer_scale:
                fc2_weight = fc2_weight * self.ls[:, None]
        return fc1_bias, fc1_weight, fc2_bias, fc2_weight, self.act_channels



class RePaMlp(nn.Module):
    def __init__(self, 
                 fc1_bias, 
                 fc1_weight, 
                 fc2_bias, 
                 fc2_weight,
                 act_channels, 
                 act_layer):
        super().__init__()
        
        dim = fc1_weight.shape[1]
        # Hyperparameters
        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, act_channels, 1)
        self.fc3 = nn.Conv2d(act_channels, dim, 1, bias=False)
        self.act = act_layer()
        
        with torch.no_grad():
            weight1 = fc1_weight[act_channels:, :].T @ fc2_weight[:, act_channels:].T + torch.eye(dim).to(fc1_weight.device)
            weight2 = fc1_weight[:act_channels, :]
            weight3 = fc2_weight[:, :act_channels] 
            bias1 = (fc1_bias[act_channels:].unsqueeze(0) @ fc2_weight[:, act_channels:].T).squeeze() + fc2_bias
            bias2 = fc1_bias[:act_channels]
            
            self.fc1.weight.copy_(weight1.T[:, :, None, None])
            self.fc1.bias.copy_(bias1)
            self.fc2.weight.copy_(weight2[:, :, None, None])
            self.fc2.bias.copy_(bias2)
            self.fc3.weight.copy_(weight3[:, :, None, None])
        
    def forward(self, x):
        x = self.fc3(self.act(self.fc2(x))) + self.fc1(x)
        return x
        
        
        
class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv. 
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0, 
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x



class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x



class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)



class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x



class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth, 
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 channel_idle=False, feature_norm="LayerNorm", idle_ratio=0.75):

        super().__init__()
        
        self.dim = dim
        self.act_layer = act_layer
        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim_in=dim, dim_hidden=mlp_hidden_dim, 
                       act_layer=act_layer, drop_path=drop_path, 
                       channel_idle=channel_idle, feature_norm=feature_norm,
                       init_values=layer_scale_init_value, layer_scale=use_layer_scale,
                       idle_ratio=idle_ratio)
        
        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = self.mlp(x)
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = self.mlp(x)
        return x
        
    def reparam(self):
        fc1_bias, fc1_weight, fc2_bias, fc2_weight, act_channels = self.mlp.reparam()
        del self.mlp
        self.mlp = RePaMlp(fc1_bias, fc1_weight, fc2_bias, fc2_weight, act_channels, self.act_layer)
        return
    
    

def basic_blocks(dim, index, layers, 
                 pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop_rate=.0, drop_path_rate=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 channel_idle=False, feature_norm="LayerNorm", idle_ratio=0.75):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks 
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PoolFormerBlock(
            dim, pool_size=pool_size, mlp_ratio=mlp_ratio, 
            act_layer=act_layer, norm_layer=norm_layer, 
            drop=drop_rate, drop_path=block_dpr, 
            use_layer_scale=use_layer_scale, 
            layer_scale_init_value=layer_scale_init_value, 
            channel_idle=channel_idle, feature_norm=feature_norm,
            idle_ratio=idle_ratio
            ))
    blocks = nn.Sequential(*blocks)

    return blocks



class PoolFormer(nn.Module):
    """
    PoolFormer, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, --pool_size: the embedding dims, mlp ratios and 
        pooling size for the 4 stages
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalization and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad: 
        specify the downsample (patch embed.)
    --fork_feat: whether output features of the 4 stages, for dense prediction
    --init_cfg, --pretrained: 
        for mmdetection and mmsegmentation to load pretrained weights
    """
    def __init__(self, layers, embed_dims=None, 
                 mlp_ratios=None, downsamples=None, 
                 pool_size=3, 
                 norm_layer=GroupNorm, act_layer=nn.GELU, 
                 num_classes=1000,
                 in_patch_size=7, in_stride=4, in_pad=2, 
                 down_patch_size=3, down_stride=2, down_pad=1, 
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5, 
                 fork_feat=False,
                 init_cfg=None, 
                 pretrained=None, 
                 channel_idle=False,
                 feature_norm="LayerNorm",
                 idle_ratio=0.75,
                 **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad, 
            in_chans=3, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, 
                                 pool_size=pool_size, mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer, 
                                 drop_rate=drop_rate, 
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale, 
                                 layer_scale_init_value=layer_scale_init_value,
                                 channel_idle=channel_idle, feature_norm=feature_norm,
                                 idle_ratio=idle_ratio)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride, 
                        padding=down_pad, 
                        in_chans=embed_dims[i], embed_dim=embed_dims[i+1]
                        )
                    )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model 
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading 
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)
            
            # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return cls_out
    
    def reparam(self):
        for block in self.network:
            if not isinstance(block, PatchEmbed):
                for blk in block:
                    blk.reparam()



model_urls = {
    "poolformer_s12": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s12.pth.tar",
    "poolformer_s24": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s24.pth.tar",
    "poolformer_s36": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s36.pth.tar",
    "poolformer_m36": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m36.pth.tar",
    "poolformer_m48": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m48.pth.tar",
}



@register_model
def RePaPoolformer_s12(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    """
    PoolFormer-S12 model, Params: 12M
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios: 
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pool_size = 7
    model = PoolFormer(
        layers, embed_dims=embed_dims, pool_size=pool_size,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        **kwargs)
    return model


@register_model
def RePaPoolformer_s24(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    """
    PoolFormer-S24 model, Params: 21M
    """
    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pool_size = 7
    model = PoolFormer(
        layers, embed_dims=embed_dims, pool_size=pool_size,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        **kwargs)
    return model


@register_model
def RePaPoolformer_s36(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    """
    PoolFormer-S36 model, Params: 31M
    """
    layers = [6, 6, 18, 6]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pool_size = 7
    model = PoolFormer(
        layers, embed_dims=embed_dims, pool_size=pool_size,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        layer_scale_init_value=1e-6, 
        **kwargs)
    return model


@register_model
def RePaPoolformer_m36(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    """
    PoolFormer-M36 model, Params: 56M
    """
    layers = [6, 6, 18, 6]
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pool_size = 7
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        layer_scale_init_value=1e-6, 
        **kwargs)
    return model


@register_model
def RePaPoolformer_m48(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    """
    PoolFormer-M48 model, Params: 73M
    """
    layers = [8, 8, 24, 8]
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pool_size = 7
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        layer_scale_init_value=1e-6, 
        **kwargs)
    return model