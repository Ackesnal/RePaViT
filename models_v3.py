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
import torch.utils.checkpoint as ckpt

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            dim_in,
            dim_hidden=None,
            dim_out=None,
            bias=False,
            drop_path=0.,
            use_conv=False,
            channel_idle=False,
            act_layer=nn.GELU,
            feature_norm="LayerNorm",
            shortcut_gain=0.0,
            std=1.0):
            
        super().__init__()
        
        ######################## ↓↓↓↓↓↓ ########################
        # Hyperparameters
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden or dim_in
        self.dim_out = dim_out or dim_in
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        # Self-attention projections
        self.ffn1 = nn.Linear(self.dim_in, self.dim_hidden, bias=bias)
        self.ffn2 = nn.Linear(self.dim_hidden, self.dim_out, bias=bias)
        self.act = act_layer()
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        self.channel_idle = channel_idle
        if self.channel_idle:
            self.gain = nn.Parameter(torch.ones((1))*shortcut_gain, 
                                     requires_grad=False)
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        self.feature_norm = feature_norm
        if self.feature_norm == "LayerNorm":
            self.norm = nn.LayerNorm(self.dim_in)
        elif self.feature_norm == "BatchNorm":
            self.norm1 = nn.BatchNorm1d(self.dim_in)
            self.norm2 = nn.BatchNorm1d(self.dim_hidden)
        elif self.feature_norm == "EmpiricalSTD":
            self.std1 = nn.Parameter(torch.ones((1))*std)
            self.std2 = nn.Parameter(torch.ones((1))*std)
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None
        ######################## ↑↑↑↑↑↑ ########################
            
    def forward(self, x):
        B, N, C = x.shape
        ######################## ↓↓↓ 2-layer MLP ↓↓↓ ########################
        shortcut = x # B, N, C
        
        # Feature normalization
        if self.feature_norm == "LayerNorm":
            x = self.norm(x)
        elif self.feature_norm == "BatchNorm":
            x = self.norm1(x.transpose(-1,-2)).transpose(-1, -2)
        elif self.feature_norm == "EmpiricalSTD":
            x = x / self.std1.unsqueeze(0).unsqueeze(-1)
        else:
            pass
        
        # FFN in
        x = self.ffn1(x) # B, N, 4C
        
        
        # Activation
        if self.channel_idle:
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask[:, :, :C] = True
            x = torch.where(mask, self.act(x), x)
        else:
            x = self.act(x)
        
        if self.feature_norm == "BatchNorm":
            x = self.norm2(x.transpose(-1,-2)).transpose(-1, -2)
        elif self.feature_norm == "EmpiricalSTD":
            x = x / self.std2.unsqueeze(0).unsqueeze(-1)
        else:
            pass
            
        # FFN out
        x = self.ffn2(x)
        
        if self.channel_idle:
            x = x * self.gain
        
        x = self.drop_path(x) if self.drop_path is not None else x
        
        x = x + shortcut
        ######################## ↑↑↑ 2-layer MLP ↑↑↑ ########################
        #if x.get_device() == 0:
            #print("x after ffn:", x.std(-1).mean().item(), x.mean().item(), x.max().item(), x.min().item())
            #print("Shortcut gain", self.gain.data.item())
        return x
        
    def reparam(self):
        
        return
        
        
        
class Attention(nn.Module):
    def __init__(self, 
                 dim, 
                 num_head=6, 
                 bias=True,
                 qk_scale=None, 
                 attn_drop=0.,
                 drop_path=0., 
                 feature_norm="LayerNorm",
                 po_shortcut=False,
                 shortcut_gain=1.0,
                 std=1.0):
                 
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
        # Per-operation shortcut
        self.po_shortcut = po_shortcut
        if self.po_shortcut:
            self.gain1 = nn.Parameter(torch.ones((1))*shortcut_gain, 
                                      requires_grad=False)
            self.gain2 = nn.Parameter(torch.ones((1))*shortcut_gain, 
                                      requires_grad=False)
            self.gain3 = nn.Parameter(torch.ones((1))*shortcut_gain, 
                                      requires_grad=False)
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None
        # Attention drop
        self.attn_drop = attn_drop
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        # Normalization
        self.feature_norm = feature_norm
        if self.feature_norm == "LayerNorm":
            if self.po_shortcut:
                self.norm1 = nn.LayerNorm(self.dim)
                self.norm2 = nn.LayerNorm(self.dim)
                self.norm3 = nn.LayerNorm(self.dim)
            else:
                self.norm = nn.LayerNorm(self.dim)
        elif self.feature_norm == "BatchNorm":
            if self.po_shortcut:
                self.norm1 = nn.BatchNorm1d(self.dim)
                self.norm2 = nn.BatchNorm1d(self.dim)
                self.norm3 = nn.BatchNorm1d(self.dim)
            else:
                self.norm = nn.BatchNorm1d(dim)
        elif self.feature_norm == "EmpiricalSTD":
            if self.po_shortcut:
                self.std1 = nn.Parameter(torch.ones((1))*std)
                self.std2 = nn.Parameter(torch.ones((1))*std)
                self.std3 = nn.Parameter(torch.ones((1))*std)
            else:
                self.std = nn.Parameter(torch.ones((1))*std)
        ######################## ↑↑↑↑↑↑ ########################
        
    def forward(self, x):
        B, N, C = x.shape
        
        if not self.po_shortcut:
            # Shortcut
            shortcut = x
            
            # Feature normalization
            if self.feature_norm == "LayerNorm":
                x = self.norm(x)
            elif self.feature_norm == "BatchNorm":
                x = self.norm(x.transpose(-1,-2)).transpose(-1,-2)
            elif self.feature_norm == "EmpiricalSTD":
                x = x / self.std.unsqueeze(0).unsqueeze(-1)
            else:
                pass
            
            # Project to QKV
            qkv = self.qkv(x)
            qkv = rearrange(qkv, 'b n (k nh hc) -> k b nh n hc', k=3, nh=self.num_head)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
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
                
        elif self.po_shortcut:
            # Shortcut 1
            shortcut = x
            
            # Feature normalization
            if self.feature_norm == "LayerNorm":
                x = self.norm1(x)
            elif self.feature_norm == "BatchNorm":
                x = self.norm1(x.transpose(-1,-2)).transpose(-1,-2)
            elif self.feature_norm == "EmpiricalSTD":
                x = x / self.std1
            else:
                pass
            
            # Project to QKV
            qkv = self.qkv(x)
            qkv = rearrange(qkv, 'b n (k nh hc) -> k b nh n hc', k=3, nh=self.num_head)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Add shortcut 1
            v = rearrange(v, 'b nh n hc -> b n (nh hc)')
            v = v * self.gain1
            v = self.drop_path(v) if self.drop_path is not None else v
            v = v + shortcut
            
            # Shortcut 2
            shortcut = v
            
            # Feature normalization
            if self.feature_norm == "LayerNorm":
                v = self.norm2(v)
            elif self.feature_norm == "BatchNorm":
                v = self.norm2(v.transpose(-1,-2)).transpose(-1,-2)
            elif self.feature_norm == "EmpiricalSTD":
                v = v / self.std2
            else:
                pass
                
            # Self-attention
            v = rearrange(v, 'b n (nh hc) -> b nh n hc', nh=self.num_head)
            x = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)
                
            # Reshape x back to input shape
            x = rearrange(x, 'b nh n hc -> b n (nh hc)')
            
            # Add shortcut 2 
            x = x * self.gain2
            x = self.drop_path(x) if self.drop_path is not None else x
            x = x + shortcut
                
            # Shortcut 3
            shortcut = x
            
            # Feature normalization
            if self.feature_norm == "LayerNorm":
                x = self.norm3(x)
            elif self.feature_norm == "BatchNorm":
                x = self.norm3(x.transpose(-1,-2)).transpose(-1,-2)
            elif self.feature_norm == "EmpiricalSTD":
                x = x / self.std3
            else:
                pass
            
            # Output linear projection
            x = self.proj(x)
                
            # Add shortcut 3 
            x = x * self.gain3
            x = self.drop_path(x) if self.drop_path is not None else x
            x = x + shortcut
        ######################### ↑↑↑ Self-attention ↑↑↑ ##########################
        #if x.get_device() == 0:
            #print("x after mhsa:", x.std(-1).mean().item(), x.mean().item(), x.max().item(), x.min().item())
            #print("Shortcut gain", self.gain1.data.item(), self.gain2.data.item(), self.gain3.data.item())
        return x
        
    def reparam(self):
        return


class RepAttention(nn.Module):
    def __init__(self, 
                 dim, 
                 num_head,
                 q_weight=None,
                 k_weight=None,
                 v_weight=None,
                 q_bias=None,
                 k_bias=None,
                 v_bias=None
                 ):
        super().__init__()
        
        # Hyperparameters
        self.num_head = num_head
        self.dim_head = dim // num_head
        self.dim = dim
        self.scale = self.dim_head ** -0.5 # scale
        
        self.qkv = nn.Linear(dim, dim*3)
        #self.ffn1 = nn.Linear(dim, dim)
        #self.ffn2 = nn.Linear(dim, dim)
        #self.ffn3 = nn.Linear(dim, dim)
        #self.act = nn.GELU()
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, N, C = x.shape
        shortcut = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_head, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Calculate self-attention
        x = nn.functional.scaled_dot_product_attention(q, k, v) # B, nh, N, C//nh
        x = rearrange(x, 'b nh n hc -> b n (nh hc)') # B, N, C
        
        # x = self.ffn3(self.act(self.ffn2(x))) + self.ffn1(x)
        return self.out(x) + shortcut


class RepMlp(nn.Module):
    def __init__(self, weights, biases):
        super().__init__()
        
        # Hyperparameters
        self.ffn1 = nn.Linear(dim, dim)
        self.ffn2 = nn.Linear(dim, dim)
        self.ffn3 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        
        self.ffn1.weight.data = weights[0]
        self.ffn1.bias.data = biases[0]
        self.ffn2.weight.data = weights[0]
        self.ffn2.bias.data = biases[0]
        self.ffn3.weight.data = weights[0]
        self.ffn3.bias.data = biases[0]

        
    def forward(self, x):
        x = self.ffn3(self.act(self.ffn2(x))) + self.ffn1(x)
        return x
        

class NFAttentionBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_head, mlp_ratio=4., bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, channel_idle=False, po_shortcut=False, 
                 feature_norm="LayerNorm", shortcut_gain=1.0, std=1.0): 
        super().__init__()
        
        dim_hidden = int(dim * mlp_ratio)
        self.rep = False
        self.dim = dim
        self.num_head = num_head
        
        if po_shortcut:
            self.attn = Attention(dim=dim, num_head=num_head, bias=bias, qk_scale=qk_scale, 
                                  attn_drop=attn_drop, drop_path=drop_path, feature_norm=feature_norm,
                                  po_shortcut=po_shortcut, shortcut_gain=shortcut_gain, std=std)
        else:
            self.attn = Attention(dim, num_head=num_head, bias=bias, qk_scale=qk_scale, 
                                  attn_drop=attn_drop, drop_path=drop_path)
        
        if channel_idle:
            self.mlp = Mlp(dim_in=dim, dim_hidden=dim_hidden, bias=bias, act_layer=act_layer, 
                           drop_path=drop_path, feature_norm=feature_norm, std=std, 
                           channel_idle=channel_idle, shortcut_gain=shortcut_gain)
        else:
            self.mlp = Mlp(dim_in=dim, dim_hidden=dim_hidden, bias=bias, 
                           act_layer=act_layer, drop_path=drop_path)
    
    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x
    
    def reparam(self):
        weights, biases = self.mlp.reparam()
        del self.mlp
        self.mlp = RepMlp(weights, biases)
        return
        """
        if self.affected_layers == "FFN":
            self.mlp = RepMlp(self.dim)
            self.attn = RepAttention(self.dim, self.num_head, None, None, None, None, None, None) #q_weight, k_weight, v_weight, q_bias, k_bias, v_bias)
            return
        elif self.affected_layers == "MHSA":
            self.attn = RepAttention(self.dim, self.num_head, None, None, None, None, None, None) #q_weight, k_weight, v_weight, q_bias, k_bias, v_bias)
            return
        elif self.affected_layers == "Both":
            self.attn = RepAttention(self.dim, self.num_head, None, None, None, None, None, None) #q_weight, k_weight, v_weight, q_bias, k_bias, v_bias)
            return
            
        #q_weight, k_weight, v_weight, q_bias, k_bias, v_bias = self.attn.reparam()
        #ffn_weight, ffn_bias = self.mlp.reparam()
        #v_weight = ffn_weight @ v_weight
        #v_bias = nn.functional.linear(v_bias.unsqueeze(0), ffn_weight, ffn_bias).squeeze()
        self.rep = True
        del self.attn
        del self.mlp
        self.attn = RepAttention(self.dim, self.num_head, None, None, None, None, None, None) #q_weight, k_weight, v_weight, q_bias, k_bias, v_bias)
        """


class NFTransformer(VisionTransformer):
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
            block_fn=NFAttentionBlock,
            feature_norm='LayerNorm',
            channel_idle=False,
            po_shortcut=False,
            shortcut_gain=0.0,):
        
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
        std = [x.item() for x in torch.logspace(start=0, end=2, steps=depth, base=2)]
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
                channel_idle=channel_idle,
                po_shortcut=po_shortcut,
                feature_norm=feature_norm,
                shortcut_gain=shortcut_gain,
                std=std[i],
            )
            for i in range(depth)])
        
        self.num_head = num_heads
        self.dim_head = embed_dim//self.num_head
        self.pre_norm = pre_norm
        
        self.feature_norm = feature_norm
        
        self.use_checkpoint = False
        
        self._init_standard_weights()
        
    def _init_standard_weights(self):
        for name, param in self.named_parameters():
            if "norm" in name:
                if "weight" in name:
                    nn.init.constant_(param, 1.0)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
            else:
                if "weight" in name:
                    trunc_normal_(param, mean=0.0, std=.02, a=-2, b=2)
                    # param.data.mul_(0.67*math.pow(12, -0.25))
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
                
            
    def reparam(self):
        for blk in self.blocks:
            blk.reparam()
            
            
        
@register_model
def RePaViT_tiny_patch16_224_layer12(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = NFTransformer(patch_size=16, embed_dim=192, depth=12, pre_norm=True,
                          num_heads=3, mlp_ratio=4, qkv_bias=True, fc_norm=False,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
    
    
@register_model
def RePaViT_small_patch16_224_layer12(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = NFTransformer(patch_size=16, embed_dim=384, depth=12, pre_norm=True,
                          num_heads=6, mlp_ratio=4, qkv_bias=True, fc_norm=False,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
@register_model
def RePaViT_base_patch16_224_layer12(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = NFTransformer(patch_size=16, embed_dim=768, depth=12, pre_norm=True,
                          num_heads=12, mlp_ratio=4, qkv_bias=True, fc_norm=False,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model