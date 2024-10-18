
from typing import Callable, Tuple, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from functools import partial
import inspect

from ..transformer import MLPBlock
from ..conv import autopad
from .normalization import *
from .DCNv4 import *
from .utils import *

import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from .DCNv4.functions import DCNv4Function
from .definitions import *

class CNA(nn.Module):
    """Classic Convolution-Normalization-Activation topology
    
    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        k (int | tuple[int, int]): Kernel size. Default: 1
        s (int | tuple[int, int]): Stride. Default: 1
        p (int | tuple[int, int], optional): Padding. Default: None for Automaticly padding.
        g (int): Groups. Default: 1
        d (int): Dilation. Default: 1
        act (callable[[], nn.Module]], optional): Activation. Default: nn.GELU
        norm (callable[[int], nn.Module]], optional): Normalization. Default: timm.models.convnext.LayerNorm2d copies

    """
    
    def __init__(self, c1: int, c2: int, 
                 k: int | Tuple[int, int]=1, s: int | Tuple[int, int]=1, p: Optional[int | Tuple[int, int]]=None, 
                 g: int=1, d: int=1,
                 act:  Optional[Callable[[],    nn.Module]]=nn.GELU, 
                 norm: Optional[Callable[[int], nn.Module]]=LayerNorm2d,
                 bias: bool=False):
        super().__init__()
        # Direct lint from yaml handle.
        if (isinstance(norm, str)): norm = eval(norm)
        if (isinstance(act,  str)): act  = eval(act)
        assert isinstance(norm, type) and isinstance(act, type)
        
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)
        self.norm = norm(c2) if norm is not None else nn.Identity()
        self.act =  act()    if act  is not None else nn.Identity()

        
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.norm(self.conv(x)))

class DWCNA(CNA):
    def __init__(self, c1: int, c2: int, 
                 k: int | Tuple[int, int]=1, s: int | Tuple[int, int]=1, p: Optional[int | Tuple[int, int]]=None, 
                 d: int=1,
                 act:  Optional[Callable[[],    nn.Module]]=nn.GELU, 
                 norm: Optional[Callable[[int], nn.Module]]=LayerNorm2d, 
                 bias: bool=False):
        super().__init__(c1, c2, k=k, s=s, p=p, g=math.gcd(c1, c2), d=d, bias=bias, act=act, norm=norm)

class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    Copies from https://github.com/sail-sg/inceptionnext/blob/main/models/inceptionnext.py

    Args:
        c (int): Number of input channels.
        ksize (int, optional): Kernel size. Default: 3
        band_ksize (int, optional): Kernel size in the band dimension. Default: 11
        branch_ratio (float, optional): Branch ratio. Default: 0.125

    References:
        [1] W. Yu, P. Zhou, S. Yan, and X. Wang, "InceptionNeXt: When Inception Meets ConvNeXt," in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle WA, USA, Jun. 17, 2024.
    """
    def __init__(self, c, ksize=3, band_ksize=11, branch_ratio=0.125):
        super().__init__()
        
        gc = int(c * branch_ratio) # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, ksize, padding=ksize//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_ksize), padding=(0, band_ksize//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_ksize, 1), padding=(band_ksize//2, 0), groups=gc)
        self.split_indexes = [c - 3 * gc, gc, gc, gc]
        
    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), 
            dim=1,
        )

class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale    
        
class ConvMLP(nn.Module):
    def __init__(
            self, c1, c2, emb=None, 
            act:  Optional[Callable[[],    nn.Module]]=nn.GELU, 
            norm: Optional[Callable[[int], nn.Module]]=LayerNorm2d,
            bias=True, 
            drop=0.):
        """ Multi-layer perceptron equivalent using 1x1 Conv for full BCHW networks.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            emb (int, optional): Number of intermediate channels. Default: None
            act (Type[nn.Module], optional): Activation function. Default: nn.GELU
            norm (Type[nn.Module], optional): Normalization layer.  Default: None
            bias (bool, optional): Whether to use bias. Default: True
            drop (float, optional): Dropout rate. Default: 0.0
        """

        super().__init__()
        emb = emb or c1
        
        # Direct lint from yaml handle.
        if (isinstance(norm, str)): norm = eval(norm)
        if (isinstance(act,  str)): act  = eval(act)

        self.fc1 = nn.Conv2d(c1, emb, kernel_size=1, bias=bias)
        self.norm = norm(emb) if norm else nn.Identity()
        self.act  = act()     if act  else nn.Identity()
        self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()
        self.fc2 =  nn.Conv2d(emb, c2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob or 0.
    
    @staticmethod
    def drop_path_f(x, drop_prob: float=0., training: bool=False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim-1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor
    
    def forward(self, x):
        return self.drop_path_f(x, self.drop_prob, self.training)
    
class SwinV2Block(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    @staticmethod
    def window_partition(x, window_size):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows
    @staticmethod
    def window_reverse(windows, window_size, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x
    class WindowAttention(nn.Module):
        r""" Window based multi-head self attention (W-MSA) module with relative position bias.
        It supports both of shifted and non-shifted window.

        Args:
            dim (int): Number of input channels.
            window_size (tuple[int]): The height and width of the window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
            attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
            proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        """

        def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,):
            super().__init__()
            self.dim = dim
            self.window_size = window_size  # Wh, Ww
            self.num_heads = num_heads

            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

            # mlp to generate continuous relative position bias
            self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                        nn.GELU(),
                                        nn.Linear(512, num_heads, bias=False))

            # get relative_coords_table
            relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
            relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
            relative_coords_table = torch.stack(
                torch.meshgrid([relative_coords_h,
                                relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                torch.abs(relative_coords_table) + 1.0) / np.log2(8)

            self.register_buffer("relative_coords_table", relative_coords_table)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(dim))
                self.v_bias = nn.Parameter(torch.zeros(dim))
            else:
                self.q_bias = None
                self.v_bias = None
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x, mask=None):
            """
            Args:
                x: input features with shape of (num_windows*B, N, C)
                mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
            """
            B_, N, C = x.shape
            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
            q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

            # cosine attention
            attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
            logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))).exp()
            attn = attn * logit_scale

            relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
            relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
            attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)
                
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = self.WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPBlock(dim, mlp_hidden_dim, act=act_layer)


    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = self.window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        if (pad_r > 0 or pad_b > 0):
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x

class DeformConv2d_v4(nn.Module):
    SUPPORT_FORMAT = {
        "channel_first": ShapeFormat.BCHW,
        "channel_last":  ShapeFormat.BHWC,
    }    
    class CenterFeatureScaleModule(nn.Module):
        def forward(self,
                    query,
                    center_feature_scale_proj_weight,
                    center_feature_scale_proj_bias):
            center_feature_scale = F.linear(query,
                                            weight=center_feature_scale_proj_weight,
                                            bias=center_feature_scale_proj_bias).sigmoid()
            return center_feature_scale

    def __init__(
            self,
            c1: int=64,
            c2: Optional[int]=None,
            kernel_size=3,
            stride=1,
            pad=None,
            dilation=1,
            group: int=None,
            offset_scale=1.0,
            data_format='channel_first',
            dw_kernel_size=None,
            center_feature_scale=False,
            remove_center=False,
            output_bias=True,
            without_pointwise=False,
            **kwargs):
        """
        DCNv4 Module
        """
        super().__init__()
        # Set group to maximum possible value with DCNv4 if no specific group given.
        if (group is None): group = c1 // 16
        
        # Set output channels to default (As same as input)
        if (c2 == None): c2 = c1
        # Check channels input and output. Output channels different with input without pointwise are ambiguous
        if ((c1 != c2) and without_pointwise):
            raise ValueError(
                f"Output channels different with input without pointwise are ambiguous.\nGiven: c1: {c1}, c2: {c2}")
        
        if c1 % group != 0:
            raise NOT_DIVISABLE(c1, group)
        assert (c1 // group) % 16 == 0, "you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation"

        if (data_format.lower() not in self.SUPPORT_FORMAT.keys()):
            raise NOT_SUPPORTED_FORMAT(self.SUPPORT_FORMAT.keys(), self.data_format)

        self.offset_scale = offset_scale
        self.channels = c1
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = autopad(kernel_size, pad, dilation)
        self.group = group
        self.group_channels = c1 // group
        self.offset_scale = offset_scale
        self.dw_kernel_size = dw_kernel_size
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)
        self.without_pointwise = without_pointwise

        self.data_format = self.SUPPORT_FORMAT[data_format]

        self.K =  group * (kernel_size * kernel_size - self.remove_center)
        
        if (dw_kernel_size is not None):
              self.offset_mask_dw = nn.Conv2d(c1, c1, dw_kernel_size, stride=1, padding=(dw_kernel_size - 1) // 2, groups=c1)
        else: self.offset_mask_dw = nn.Identity()
        
        self.offset_mask = nn.Conv2d(c1, int(math.ceil((self.K * 3)/8)*8), 1)
        
        if not without_pointwise:
            self.value_proj  = nn.Conv2d(c1, c1, 1)
            self.output_proj = nn.Conv2d(c1, c2, 1, bias=output_bias)
        else:             
            self.value_proj  = nn.Identity()
            self.output_proj = nn.Identity()

        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, c1)))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0).view((1,)).repeat(group, ))
            self.center_feature_scale_module = self.CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset_mask.weight.data, 0.)
        constant_(self.offset_mask.bias.data, 0.)
        if not self.without_pointwise:
            xavier_uniform_(self.value_proj.weight.data)
            constant_(self.value_proj.bias.data, 0.)
            xavier_uniform_(self.output_proj.weight.data)
            if self.output_proj.bias is not None:
                constant_(self.output_proj.bias.data, 0.)

    def forward(self, x):
        # nn.Conv2d requires BCHW shapes.
        x = format_convert(x, self.data_format, ShapeFormat.BCHW)
        x_proj = self.value_proj(x)
        offset_mask_input = self.offset_mask_dw(x)
        offset_mask = self.offset_mask(offset_mask_input)

        # Deformable Conv v4 requires BHWC shapes.
        x_proj =      format_convert(x_proj     , ShapeFormat.BCHW, ShapeFormat.BHWC)
        offset_mask = format_convert(offset_mask, ShapeFormat.BCHW, ShapeFormat.BHWC)
        x = DCNv4Function.apply(
            x_proj, offset_mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center
            )

        if (self.center_feature_scale):
            center_feature_scale = self.center_feature_scale_module(
                x, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale

        # Convert back before Pointwise Conv/Return.
        x = format_convert(x, ShapeFormat.BHWC, ShapeFormat.BCHW)
        x = self.output_proj(x)
        
        # Convert to original input shape.
        return format_convert(x, ShapeFormat.BCHW, self.data_format)
        
class DarknetBottleneck(nn.Module):
    """Standard bottleneck with activation and normalization exposed."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, act=nn.SiLU, norm=nn.BatchNorm2d):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CNA(c1, c_, k[0], 1     , act=act, norm=norm)
        self.cv2 = CNA(c_, c2, k[1], 1, g=g, act=act, norm=norm)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    

class SubPixelConv(nn.Module):
    """
    Sub-Pixel Convolution
    
    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        upscale_factor (int, optional): Upscale factor. Defaults to 2.
        kernel_size (int, optional): Convolution kernel size. Defaults to 3.
        act (callable[[], nn.Module]], optional): Activation. Defaults to `nn.GELU`
        norm (callable[[int], nn.Module]], optional): Normalization. Defaults to `timm.models.convnext.LayerNorm2d` copies
        
    Referencez:
        [1] W. Shi, J. Caballero, F. Huszár, J. Totz, A. P. Aitken, R. Bishop, D. Rueckert, and Z. Wang. "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" in Proc. CVPR 2016. 
    """
    def __init__(self, c1: int, c2: int, 
                 upscale_factor: int=2, 
                 kernel_size: int=3, 
                 act:  Optional[Callable[[],    nn.Module]]=nn.GELU, 
                 norm: Optional[Callable[[int], nn.Module]]=LayerNorm2d,):
        
        super().__init__()
        self.c2 = c2 if c2 > 0 else c1
        self.r = upscale_factor
        self.cv1 = CNA(c1, c2 * (upscale_factor ** 2), k=kernel_size, act=act, norm=norm)
        self.upsample = nn.PixelShuffle(self.r)
    def forward(self, x: Tensor) -> Tensor:
        x = self.cv1(x)
        x = self.upsample(x)
        return x

class DeformedSubPixelConv(nn.Module):
    def __init__(self, c1: int, c2: int, 
                 upscale_factor: int=2, 
                 kernel_size: int=3,
                 act=nn.GELU, norm=LayerNorm2d):
        super().__init__()
        self.c2 = c2 if c2 > 0 else c1
        self.r = r = upscale_factor
        self._cr =  c2 * (r ** 2)
        self._act   = act()             if act  is not None else nn.Identity()
        self._norm  = norm(self._cr)    if norm is not None else nn.Identity()
        self.cv1 = DWCNA(c1, self._cr, k=kernel_size, act=act, norm=norm)
        self.cv2 = DCNv4(self._cr, self._cr, k=kernel_size, group=int((self._cr) // 16))
        self.upsample = nn.PixelShuffle(self.r)

    def forward(self, x: Tensor) -> Tensor:
        x = self.cv1(x)
        _x = self.cv2(x)
        _x = self._norm(_x)
        x = x + _x
        x = self._act(x)
        
        x = self.upsample(x)
        return x

