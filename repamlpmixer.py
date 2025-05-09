# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
import torch.autograd.profiler as profiler

from timm.models.mlp_mixer import MlpMixer, _cfg
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
            channel_idle=False,
            act_layer=nn.GELU,
            feature_norm="LayerNorm"):
            
        super().__init__()
        
        ######################## ↓↓↓↓↓↓ ########################
        # Hyperparameters
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden or dim_in
        self.dim_out = dim_out or dim_in
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        # Self-attention projections
        self.fc1 = nn.Linear(self.dim_in, self.dim_hidden)
        self.fc2 = nn.Linear(self.dim_hidden, self.dim_out)
        self.act = act_layer()
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        # Channel-idle
        self.channel_idle = channel_idle
        self.act_channels = dim_in
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        self.feature_norm = feature_norm
        if self.feature_norm == "LayerNorm":
            self.norm = nn.LayerNorm(self.dim_in)
        elif self.feature_norm == "BatchNorm":
            self.norm1 = nn.BatchNorm1d(self.dim_in)
            self.norm2 = nn.BatchNorm1d(self.dim_hidden)
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None
        ######################## ↑↑↑↑↑↑ ########################
        
    def forward(self, x):
        B, N, C = x.shape
        ######################## ↓↓↓ 2-layer MLP ↓↓↓ ########################
        shortcut = x # B, N, C
        
        # 1st Feature normalization
        if self.feature_norm == "LayerNorm":
            x = self.norm(x)
        elif self.feature_norm == "BatchNorm":
            x = self.norm1(x.transpose(-1,-2)).transpose(-1, -2)
        
        # FFN in
        x = self.fc1(x) # B, N, 4C
        
        # Activation
        if self.channel_idle:
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask[:, :, :self.act_channels] = True
            x = torch.where(mask, self.act(x), x)
        else:
            x = self.act(x)
        
        # 2nd Feature normalization
        if self.feature_norm == "BatchNorm":
            x = self.norm2(x.transpose(-1,-2)).transpose(-1, -2)
            
        # FFN out
        x = self.fc2(x)
        
        # Add DropPath
        x = self.drop_path(x) if self.drop_path is not None else x
        
        x = x + shortcut
        ######################## ↑↑↑ 2-layer MLP ↑↑↑ ########################
        #if x.get_device() == 0:
            #print("x after ffn:", x.std(-1).mean().item(), x.mean().item(), x.max().item(), x.min().item())
        return x
        
    def reparam(self):
        self.eval()
        with torch.no_grad():
            mean = self.norm1.running_mean
            std = torch.sqrt(self.norm1.running_var + self.norm1.eps)
            weight = self.norm1.weight
            bias = self.norm1.bias
            
            fc1_bias = self.fc1(-mean/std*weight+bias)
            fc1_weight = self.fc1.weight / std[None, :] * weight[None, :]
            
            mean = self.norm2.running_mean
            std = torch.sqrt(self.norm2.running_var + self.norm2.eps)
            weight = self.norm2.weight
            bias = self.norm2.bias
            
            fc2_bias = self.fc2(-mean/std*weight+bias)
            fc2_weight = self.fc2.weight / std[None, :] * weight[None, :]
        
        return fc1_bias, fc1_weight, fc2_bias, fc2_weight
        
        

class RePaMlp(nn.Module):
    def __init__(self, 
                 fc1_bias, 
                 fc1_weight, 
                 fc2_bias, 
                 fc2_weight, 
                 act_layer):
        super().__init__()
        
        dim = fc1_weight.shape[1]
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim, bias=False)
        self.act = act_layer()
        
        with torch.no_grad():
            weight1 = fc1_weight[dim:, :].T @ fc2_weight[:, dim:].T + torch.eye(dim).to(fc1_weight.device)
            weight2 = fc1_weight[:dim, :]
            weight3 = fc2_weight[:, :dim] 
            bias1 = (fc1_bias[dim:].unsqueeze(0) @ fc2_weight[:, dim:].T).squeeze() + fc2_bias
            bias2 = fc1_bias[:dim]
            
            self.fc1.weight.copy_(weight1.T)
            self.fc1.bias.copy_(bias1)
            self.fc2.weight.copy_(weight2)
            self.fc2.bias.copy_(bias2)
            self.fc3.weight.copy_(weight3)
        
    def forward(self, x):
        with torch.no_grad():
            x = self.fc3(self.act(self.fc2(x))) + self.fc1(x)
            return x
            
            

class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self,
            dim,
            seq_len,
            mlp_ratio=(0.5, 4.0),
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop=0.,
            drop_path=0.,
            channel_idle=False,
            feature_norm="LayerNorm"
    ):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        
        self.act_layer = act_layer
        
        self.mlp_tokens = mlp_layer(dim_in=seq_len, dim_hidden=tokens_dim, act_layer=act_layer, drop_path=drop_path)
        self.mlp_channels = mlp_layer(dim_in=dim, dim_hidden=channels_dim, act_layer=act_layer, drop_path=drop_path,
                                      feature_norm=feature_norm, channel_idle=channel_idle)

    def forward(self, x):
        x = self.mlp_tokens(x.transpose(1, 2)).transpose(1, 2)
        x = self.mlp_channels(x)
        return x
    
    def reparam(self):
        fc1_bias, fc1_weight, fc2_bias, fc2_weight = self.mlp_channels.reparam()
        del self.mlp_channels
        self.mlp_channels = RePaMlp(fc1_bias, fc1_weight, fc2_bias, fc2_weight, self.act_layer)
        return
        
        
        
class RePaMlpMixer(MlpMixer):

    def __init__(
            self,
            num_classes=1000,
            img_size=224,
            in_chans=3,
            patch_size=16,
            num_blocks=8,
            embed_dim=512,
            mlp_ratio=(0.5, 4.0),
            block_layer=MixerBlock,
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop_rate=0.,
            proj_drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False,
            stem_norm=False,
            global_pool='avg',
            feature_norm='LayerNorm',
            channel_idle=False, **kwargs):
        super().__init__(num_classes=num_classes,
                         img_size=img_size,
                         in_chans=in_chans,
                         patch_size=patch_size,
                         num_blocks=num_blocks,
                         embed_dim=embed_dim,
                         mlp_ratio=mlp_ratio,
                         block_layer=block_layer,
                         mlp_layer=mlp_layer,
                         norm_layer=norm_layer,
                         act_layer=act_layer,
                         drop_rate=drop_rate,
                         proj_drop_rate=proj_drop_rate,
                         drop_path_rate=drop_path_rate,
                         nlhb=nlhb,
                         stem_norm=stem_norm,
                         global_pool=global_pool)
                         
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.grad_checkpointing = False
        
        self.stem = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if stem_norm else None,
        )
        
        reduction = self.stem.feat_ratio() if hasattr(self.stem, 'feat_ratio') else patch_size
        # FIXME drop_path (stochastic depth scaling rule or all the same?)
        self.blocks = nn.Sequential(*[
            block_layer(
                embed_dim,
                self.stem.num_patches,
                mlp_ratio,
                mlp_layer=mlp_layer,
                norm_layer=norm_layer,
                act_layer=act_layer,
                drop=proj_drop_rate,
                drop_path=drop_path_rate,
                channel_idle=channel_idle,
                feature_norm=feature_norm
            )
            for _ in range(num_blocks)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=reduction) for i in range(num_blocks)]
        self.norm = norm_layer(embed_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(nlhb=nlhb)
        
    def reparam(self):
        for blk in self.blocks:
            blk.reparam()



@register_model
def RePaMlpMixer_s32(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    """ Mixer-S/32 224x224
    Paper: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model = RePaMlpMixer(patch_size=32, num_blocks=8, embed_dim=512, **kwargs)
    return model


@register_model
def RePaMlpMixer_s16(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    """ Mixer-S/16 224x224
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model = RePaMlpMixer(patch_size=16, num_blocks=8, embed_dim=512, **kwargs)
    return model


@register_model
def RePaMlpMixer_b32(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    """ Mixer-B/32 224x224
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model = RePaMlpMixer(patch_size=32, num_blocks=12, embed_dim=768, **kwargs)
    return model


@register_model
def RePaMlpMixer_b16(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    """ Mixer-B/16 224x224. ImageNet-1k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model = RePaMlpMixer(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    return model
    
    
@register_model
def RePaMlpMixer_l16(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    """ Mixer-B/16 224x224. ImageNet-1k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model = RePaMlpMixer(patch_size=16, num_blocks=24, embed_dim=1024, **kwargs)
    return model
    
    
@register_model
def RePaMlpMixer_l32(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    """ Mixer-B/16 224x224. ImageNet-1k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model = RePaMlpMixer(patch_size=32, num_blocks=24, embed_dim=1024, **kwargs)
    return model