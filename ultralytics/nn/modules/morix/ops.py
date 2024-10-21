
from typing import Callable, Optional, Tuple, Type, Union
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
from ....morix.utils.formatting import *

import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from .DCNv4.functions import DCNv4Function
from .definitions import *

class CNA(nn.Module):
    """Classic Convolution-Normalization-Activation topology
    
    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        k (Union[int, Tuple[int, int]]): Kernel size. Default: 1
        s (Union[int, Tuple[int, int]]): Stride. Default: 1
        p (Union[int, Tuple[int, int]], optional): Padding. Default: None for Automaticly padding.
        g (int): Groups. Default: 1
        d (int): Dilation. Default: 1
        act (callable[[], nn.Module]], optional): Activation. Default: nn.GELU
        norm (callable[[int], nn.Module]], optional): Normalization. Default: timm.models.convnext.LayerNorm2d copies

    """
    
    def __init__(self, c1: int, c2: int, 
                 k: Union[int, Tuple[int, int]]=1, s: Union[int, Tuple[int, int]]=1, p: Optional[Union[int, Tuple[int, int]]]=None, 
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

class TCNA(CNA):
    
    """Alias of Convolution-Normalization-Activation topology with Transposed Convolution.
    
    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        k (Union[int, Tuple[int, int]]): Kernel size. Default: 1
        s (Union[int, Tuple[int, int]]): Stride. Default: 1
        p (Union[int, Tuple[int, int]], optional): Padding. Default: None for Automaticly padding.
        g (int): Groups. Default: 1
        d (int): Dilation. Default: 1
        act (callable[[], nn.Module]], optional): Activation. Default: nn.GELU
        norm (callable[[int], nn.Module]], optional): Normalization. Default: timm.models.convnext.LayerNorm2d copies

    """
    def __init__(self, c1: int, c2: int, 
                 k: Union[int, Tuple[int, int]]=1, s: Union[int, Tuple[int, int]]=1, p: Optional[Union[int, Tuple[int, int]]]=None, 
                 g: int=1, d: int=1,
                 act:  Optional[Callable[[],    nn.Module]]=nn.GELU, 
                 norm: Optional[Callable[[int], nn.Module]]=LayerNorm2d,
                 bias: bool=False):
        super().__init__(c1, c2, k=k, s=s, p=p, g=g, d=d, act=act, norm=norm, bias=bias)
        self.conv = nn.ConvTranspose2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)
           
class DWCNA(CNA):
    """Alias of Convolution-Normalization-Activation topology with Depth-Wise Convolution.
    
    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        k (Union[int, Tuple[int, int]]): Kernel size. Default: 1
        s (Union[int, Tuple[int, int]]): Stride. Default: 1
        p (Union[int, Tuple[int, int]], optional): Padding. Default: None for Automaticly padding.
        g (int): Groups. Default: 1
        d (int): Dilation. Default: 1
        act (callable[[], nn.Module]], optional): Activation. Default: nn.GELU
        norm (callable[[int], nn.Module]], optional): Normalization. Default: timm.models.convnext.LayerNorm2d copies

    """
    def __init__(self, c1: int, c2: int, 
                 k: Union[int, Tuple[int, int]]=1, s: Union[int, Tuple[int, int]]=1, p: Optional[Union[int, Tuple[int, int]]]=None, 
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
        x = ShapeFormatHelper.format_convert(x, self.data_format, ShapeFormat.BCHW)
        x_proj = self.value_proj(x)
        offset_mask_input = self.offset_mask_dw(x)
        offset_mask = self.offset_mask(offset_mask_input)

        # Deformable Conv v4 requires BHWC shapes.
        x_proj =      ShapeFormatHelper.format_convert(x_proj     , ShapeFormat.BCHW, ShapeFormat.BHWC)
        offset_mask = ShapeFormatHelper.format_convert(offset_mask, ShapeFormat.BCHW, ShapeFormat.BHWC)
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
        x = ShapeFormatHelper.format_convert(x, ShapeFormat.BHWC, ShapeFormat.BCHW)
        x = self.output_proj(x)
        
        # Convert to original input shape.
        return ShapeFormatHelper.format_convert(x, ShapeFormat.BCHW, self.data_format)
        
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

class SubPixelDWConv(nn.Module):
    def __init__(self, c1: int, c2: int, 
                upscale_factor: int=2, 
                kernel_size: int=3, 
                act:  Optional[Callable[[],    nn.Module]]=nn.GELU, 
                norm: Optional[Callable[[int], nn.Module]]=LayerNorm2d,):
        super().__init__()
        self.c2 = c2 if c2 > 0 else c1
        self.r = upscale_factor
        self.cv1 = DWCNA(c1, c2 * (upscale_factor ** 2), k=kernel_size, act=act, norm=norm)
        self.upsample = nn.PixelShuffle(self.r)
    def forward(self, x: Tensor) -> Tensor:
        x = self.cv1(x)
        x = self.upsample(x)
        return x
