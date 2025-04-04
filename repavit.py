# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from einops import rearrange, reduce

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models._registry import register_model
from timm.layers import DropPath, trunc_normal_, PatchEmbed
import math
import torch.autograd.profiler as profiler



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
        self.fc1 = nn.Linear(self.dim_in, self.dim_hidden, bias=bias)
        self.fc2 = nn.Linear(self.dim_hidden, self.dim_out, bias=bias)
        self.act = act_layer()
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        # Channel-idle
        self.channel_idle = channel_idle
        self.act_channels = math.ceil(dim_hidden * (1-idle_ratio))
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
        self.input = x
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
            if self.act_channels == 0:
                pass
            elif self.act_channels < self.dim_hidden:
                mask = torch.zeros_like(x, dtype=torch.bool)
                mask[:, :, :self.act_channels] = True
                x = torch.where(mask, self.act(x), x)
            else:
                x = self.act(x)
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
        # ######################## ↑↑↑ 2-layer MLP ↑↑↑ ########################
        # if x.get_device() == 0:
        #     print("x grad after ffn:", self.fc2.weight.grad)
            # print("x after ffn:", x.std(-1).mean().item(), x.mean().item(), x.max().item(), x.min().item())
        return x
        
    def reparam(self):
        self.eval()
        with torch.no_grad():
            mean = self.norm1.running_mean
            std = torch.sqrt(self.norm1.running_var + self.norm1.eps)
            weight = self.norm1.weight
            bias = self.norm1.bias
            
            fc1_bias = self.fc1((-mean) / std * weight + bias)
            fc1_weight = self.fc1.weight / std[None, :] * weight[None, :]
            
            mean = self.norm2.running_mean
            std = torch.sqrt(self.norm2.running_var + self.norm2.eps)
            weight = self.norm2.weight
            bias = self.norm2.bias
            
            fc2_bias = self.fc2((-mean) / std * weight + bias)
            fc2_weight = self.fc2.weight / std[None, :] * weight[None, :]
        
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
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, act_channels)
        self.fc3 = nn.Linear(act_channels, dim, bias=False)
        self.act = act_layer()
        self.act_channels = act_channels
        
        with torch.no_grad():
            if act_channels == 0:
                weight1 = fc1_weight.T @ fc2_weight.T + torch.eye(dim).to(fc1_weight.device)
                bias1 = (fc1_bias.unsqueeze(0) @ fc2_weight.T).squeeze() + fc2_bias
                self.fc1.weight.copy_(weight1.T)
                self.fc1.bias.copy_(bias1)
                del self.fc2
                del self.fc3
                del self.act
            else:
                weight1 = fc1_weight[act_channels:, :].T @ fc2_weight[:, act_channels:].T + torch.eye(dim).to(fc1_weight.device)
                weight2 = fc1_weight[:act_channels, :]
                weight3 = fc2_weight[:, :act_channels] 
                bias1 = (fc1_bias[act_channels:].unsqueeze(0) @ fc2_weight[:, act_channels:].T).squeeze() + fc2_bias
                bias2 = fc1_bias[:act_channels]
                
                self.fc1.weight.copy_(weight1.T)
                self.fc1.bias.copy_(bias1)
                self.fc2.weight.copy_(weight2)
                self.fc2.bias.copy_(bias2)
                self.fc3.weight.copy_(weight3)
        
    def forward(self, x):
        if self.act_channels == 0:
            x = self.fc1(x)
        else:
            x = self.fc3(self.act(self.fc2(x))) + self.fc1(x)
        return x
                    
        
        
class Attention(nn.Module):
    def __init__(self, 
                 dim, 
                 num_head=6, 
                 bias=True,
                 qk_scale=None, 
                 attn_drop=0.,
                 drop_path=0.,):
                 
        super().__init__()
        
        ######################## ↓↓↓↓↓↓ ########################
        # Hyperparameters
        self.num_head = num_head
        self.dim_head = dim // num_head
        self.dim = dim
        self.scale = qk_scale or self.dim_head ** -0.5 # scale
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        # Self-attention projections
        self.qkv = nn.Linear(self.dim, 3*self.dim, bias=bias)
        self.proj = nn.Linear(self.dim, self.dim, bias=bias)
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None
        # Attention drop
        self.attn_drop = attn_drop
        ######################## ↑↑↑↑↑↑ ########################
        
        self.norm = nn.LayerNorm(self.dim)
            
    def forward(self, x):        
        # Shortcut
        shortcut = x
            
        # Feature normalization
        x = self.norm(x)
            
        # Project to QKV
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n (k nh hc) -> k b nh n hc', k=3, nh=self.num_head)
        q, k, v = qkv.unbind()
            
        # Self-attention
        x = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)
                
        # Reshape x back to input shape
        x = rearrange(x, 'b nh n hc -> b n (nh hc)')
                
        # Output linear projection
        x = self.proj(x)
        
        # Add DropPath
        x = self.drop_path(x) if self.drop_path is not None else x
            
        # Add shortcut
        x = x + shortcut
                
        return x

        

class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_head, mlp_ratio=4., bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, channel_idle=False, idle_ratio=0.75, feature_norm="LayerNorm"): 
        super().__init__()
        
        dim_hidden = int(dim * mlp_ratio)
        self.rep = False
        self.dim = dim
        self.num_head = num_head
        self.act_layer = act_layer
        
        self.attn = Attention(dim, num_head=num_head, bias=bias, qk_scale=qk_scale, 
                              attn_drop=attn_drop, drop_path=drop_path)
        
        if channel_idle:
            self.mlp = Mlp(dim_in=dim, dim_hidden=dim_hidden, bias=bias, act_layer=act_layer, 
                           drop_path=drop_path, feature_norm=feature_norm, 
                           channel_idle=channel_idle, idle_ratio=idle_ratio)
        else:
            self.mlp = Mlp(dim_in=dim, dim_hidden=dim_hidden, bias=bias, 
                           act_layer=act_layer, drop_path=drop_path)
    
    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x
    
    def reparam(self, layer):
        print(f"Layer {layer} reparamed")
        fc1_bias, fc1_weight, fc2_bias, fc2_weight, act_channels = self.mlp.reparam()
        del self.mlp
        self.mlp = RePaMlp(fc1_bias, fc1_weight, fc2_bias, fc2_weight, act_channels, self.act_layer)
        return



class Transformer(VisionTransformer):
    def __init__(self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            global_pool='token',
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=2.,
            qkv_bias=True,
            layer_scale=False,
            init_values=None,
            class_token=True,
            no_embed_class=False,
            pre_norm=False,
            fc_norm=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            weight_init='',
            embed_layer=PatchEmbed,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            block_fn=Block,
            feature_norm='LayerNorm',
            channel_idle=False,
            idle_ratio=0.75,
            heuristic="static",
            **kwargs):
        
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            class_token=class_token,
            no_embed_class=no_embed_class,
            pre_norm=pre_norm,
            fc_norm=fc_norm,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate)
            
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        if heuristic.strip().lower() in ["none", "static", ""]:
            idle_ratio = [idle_ratio for i in range(depth)]
        elif heuristic.strip().lower() in ["linear"]:
            idle_ratio = [(2*idle_ratio-1) + (2-2*idle_ratio) / (depth-1) * (i+1) for i in range(depth-1)] + [0.0]
        print("Idle Ratios:", idle_ratio)
        
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_head=num_heads,
                mlp_ratio=mlp_ratio,
                bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                act_layer=act_layer,
                feature_norm=feature_norm,
                channel_idle=channel_idle,
                idle_ratio=idle_ratio[i]
            )
            for i in range(depth)])
        
        self.num_head = num_heads
        self.dim_head = embed_dim//self.num_head
        self.pre_norm = pre_norm
        self.init_weights()
        
    def reparam(self):
        for layer, blk in enumerate(self.blocks):
            blk.reparam(layer)
            
        
@register_model
def RePaViT_Tiny(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = Transformer(patch_size=16, embed_dim=192, depth=12, pre_norm=True,
                        num_heads=3, mlp_ratio=4, qkv_bias=True, fc_norm=False,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
    
@register_model
def RePaViT_Small(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = Transformer(patch_size=16, embed_dim=384, depth=12, pre_norm=True,
                        num_heads=6, mlp_ratio=4, qkv_bias=True, fc_norm=False,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

    
@register_model
def RePaViT_Base(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = Transformer(patch_size=16, embed_dim=768, depth=12, pre_norm=True,
                        num_heads=12, mlp_ratio=4, qkv_bias=True, fc_norm=False,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
    
@register_model
def RePaViT_Large(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = Transformer(patch_size=16, embed_dim=1024, depth=24, pre_norm=True,
                        num_heads=16, mlp_ratio=4, qkv_bias=True, fc_norm=False,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

    
@register_model
def RePaViT_Huge(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = Transformer(patch_size=16, embed_dim=1280, depth=32, pre_norm=True,
                        num_heads=16, mlp_ratio=4, qkv_bias=True, fc_norm=False,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model