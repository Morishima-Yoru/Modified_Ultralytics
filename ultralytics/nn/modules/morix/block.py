import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from ..block import ResNetBlock, Bottleneck
from .ops import *
from .wrapper import *


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        norm (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        norm_first (bool, optional): Decide the reduction operation order. True for Swin Transformer v1. Default: False
    """

    def __init__(self, c1, c2, norm=nn.LayerNorm, norm_first=False):
        super().__init__()
        assert c2 == (2 * c1)
        self.c1 = c1
        self.norm_first = norm_first
        self.reduction = nn.Linear(4 * c1, c2, bias=False)
        self.norm = norm(c2)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()

        f_pad = (H % 2 == 1) or (W % 2 == 1)
        if (f_pad):
            x = F.pad(x, (0, 0, 0, W%2, 0, H%2))
        
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        if (self.norm_first): x = self.norm(x)
        x = self.reduction(x)
        if (not self.norm_first): x = self.norm(x)

        x = x.view(B, int(H/2), int(W/2), C*2)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        c1 (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, c1=3, embed_dim=96, patch_size=4, norm=None):
        super().__init__()
        self.patch_size = (patch_size, patch_size)

        self.in_chans = c1
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(c1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm(embed_dim) if norm else None

    def forward(self, x):
        _, _, H, W = x.shape
        f_pad = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if f_pad:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))
        x = self.proj(x)
        if (self.norm is None): return x
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm(x)
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class Patchify(nn.Module):
    r""" Patch Merging Layer (Called Patchify Stem in ConvNeXt) for full Convolutional network style.
    
    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        patch_sz (int, optional): Number of patch size. Default: 4
        norm (callable[[int], nn.Module]], optional): Normalization. Default: timm.models.convnext.LayerNorm2d copies

        [1] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), New Orleans, LA, USA, pp. 11976-11986, Jun. 21, 2022.
    """

    def __init__(self, c1, emb, patch_size=2, norm_first=True, norm: Type[nn.Module]=FakeLayerNorm2d):
        super().__init__()
        
        if (isinstance(norm, str)): norm = eval(norm)
        
        self.cv1 = nn.Conv2d(c1, emb, patch_size, patch_size)
        nch = c1 if norm_first else emb
        self.norm_first = norm_first
        self.norm = norm(nch) if norm is not None else nn.Identity()

    def forward(self, x):
        return self.cv1(self.norm(x)) if self.norm_first else self.norm(self.cv1(x))
    
class ConvNeXt_Block(MetaNeXt):
    def __init__(self, c, drop_path=0., ls=1e-6, act=nn.GELU, norm=FakeLayerNorm2d):
        """
        Args:
            c (int): Number of channels.
            drop_path (float, optional): Drop path probability. Default: 0.
            ls (float, optional): Label smoothing value. Default: 1e-6.
        """
        
        super().__init__(c, 4, drop_path, ls, act=act, norm=norm)
        self.build()
    
    def token_mixer(self, c: int) -> nn.Module:
        return nn.Conv2d(c, c, kernel_size=7, padding=3, groups=c)
    
class InceptionNeXt_Block(MetaNeXt):
    def __init__(self, 
            c,
            mlp_ratio=4,
            drop_path: float=0.,
            ls: float=1e-6,
            act: Type[nn.Module]=nn.GELU,
            norm: Type[nn.Module]=LayerNorm2d,
            ):
        """
        Args:
            c (int): Number of channels.
            mlp_ratio (int, optional): MLP ratio. Default: 4.
            drop_path (float, optional): Drop path probability. Default: 0.
            ls (float, optional): Label smoothing value. Default: 1e-6.
        """
        super().__init__(c, mlp_ratio=mlp_ratio, drop_path=drop_path, init_ls=ls, act=act, norm=norm)
        self.build()
    
    def token_mixer(self, c: int) -> nn.Module:
        return InceptionDWConv2d(c)

class GELAN_InceptionNeXt(GELANWrapper):
    def __init__(self, c1, c2, n, g, mlp_ratio, transition=True, e=0.5, act=nn.GELU, norm=FakeLayerNorm2d):
        """
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of repetitions.
            g (int): Number of groups.
            mlp_ratio (int): MLP ratio.
            transition (bool, optional): Whether to use transition layer. Default: True.
            e (float, optional): Expansion ratio. Default: 0.5.
            act (nn.Module, optional): Activation function. Default: nn.GELU.
            norm (nn.Module, optional): Normalization layer. Default: FakeLayerNorm2d.
        """
        super().__init__(c1, c2, n, g, transition, e, act, norm)
        self.mlp_ratio = mlp_ratio
        self.build()
    
    def computational(self, c) -> nn.Module:
        return InceptionNeXt_Block(c, self.mlp_ratio)

class GELAN_ConvNeXt(GELANWrapper):
    def __init__(self, c1, c2, n, g, transition=True, e=0.5, act=nn.GELU, norm=FakeLayerNorm2d):
        """
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of repetitions.
            g (int): Number of groups.
            transition (bool, optional): Whether to use transition layer. Default: True.
            e (float, optional): Expansion ratio. Default: 0.5.
            act (nn.Module, optional): Activation function. Default: nn.GELU.
            norm (nn.Module, optional): Normalization layer. Default: FakeLayerNorm2d.
        """
        super().__init__(c1, c2, n, g, transition, e, act, norm)
        self.build()
    
    def computational(self, c) -> nn.Module:
        return ConvNeXt_Block(c)
    
class ConvNeXtStage(Sequentially):
    def __init__(self, c1, c2, n, act=nn.GELU, norm=FakeLayerNorm2d):
        """
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of repetitions.
        """
        super().__init__(c1, c2, n)
        self.act = act
        self.norm = norm
        self.build()

    def computational(self, c) -> nn.Module:
        return ConvNeXt_Block(c, act=self.act, norm=self.norm)
    
class InceptionNeXtStage(Sequentially):
    def __init__(self, c1, c2, n, mlp_ratio=4, act=nn.GELU, norm=LayerNorm2d):
        """
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of repetitions.
            mlp_ratio (int, optional): MLP ratio. Defaults to 4.
        """
        super().__init__(c1, c2, n)
        self.act = act
        self.norm = norm
        self.mlp_ratio = mlp_ratio
        self.build()

    def computational(self, c) -> nn.Module:
        return InceptionNeXt_Block(c, mlp_ratio=self.mlp_ratio, act=self.act, norm=self.norm)

class ELAN(GELANWrapper):
    def __init__(self, c1, c2, n, g, transition=False, e=0.5, act=nn.GELU, norm=nn.BatchNorm2d):
        """
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of repetitions.
            g (int): Number of groups.
            transition (bool, optional): Whether to use transition layer. Default: True.
            e (float, optional): Expansion ratio. Default: 0.5.
            act (nn.Module, optional): Activation function. Default: nn.GELU.
            norm (nn.Module, optional): Normalization layer. Default: nn.BatchNorm2d.
        """
        super().__init__(c1, c2, n, g, transition, e, act, norm)
        self.build()
    
    def computational(self, c) -> nn.Module:
        return CNA(c, c, 3, 1, act=self.act, norm=self.norm)

class ELAN_DarknetBottleneck(GELANWrapper):
    def __init__(self, c1, c2, n, g=1, transition=False, e=0.5, act=nn.SiLU, norm=nn.BatchNorm2d):
        super().__init__(c1, c2, n, g, transition, e, act, norm)
        self.build()
        
    def computational(self, c) -> nn.Module:
        return DarknetBottleneck(c, c, True, e=1., act=self.act, norm=self.norm)

class DCNFormer(MetaFormer):
    def __init__(self, c, mlp_ratio=4, dcn_group: int=None,
                 drop: float = 0, drop_path: float = 0, 
                 layer_scale_init_value: float = None, 
                 res_scale_init_value: float = None, 
                 norm: Type[nn.Module] = LayerNorm2d, 
                 act: Type[nn.Module] = nn.GELU):
        super().__init__(c, 
                         mlp_ratio, 
                         drop, 
                         drop_path, 
                         layer_scale_init_value, 
                         res_scale_init_value, 
                         norm, 
                         act)
        self.dcn_g = dcn_group
        self.build()
        
    def token_mixer_layer(self, c: int) -> nn.Module:
        return DeformConv2d_v4(c, c, 3, group=self.dcn_g, dw_kernel_size=3, without_pointwise=True)
  
class GELAN_DCNv4(GELANWrapper):
    def __init__(self, c1, c2, n=2, g=2, dcn_g=None, transition=True, act=nn.GELU, norm=LayerNorm2d, e=0.5):
        super().__init__(c1, c2, n, g, transition, e, act, norm)
        self.dcn_g = int((self._c) // 16) if dcn_g is None else dcn_g
        self.build()
    
    def computational(self, c) -> nn.Module:
        return nn.Sequential(
            DeformConv2d_v4(c, c, 3, 1, group=self.dcn_g, dw_kernel_size=3),
            self.norm(c), 
            self.act()
        )

class GELAN_DCNFormer(GELANWrapper):
    def __init__(self, c1, c2, n=2, g=2, dcn_group=None, transition=True,
                 mlp_ratio: int=4, drop: float=0, drop_path: float=0, 
                 layer_scale_init_value: float=None, 
                 res_scale_init_value: float=None, 
                 act=nn.GELU, norm=LayerNorm2d,
                 e=0.5, ):
        super().__init__(c1, c2, n, g, transition, e, act, norm)
        self.dcn_g = dcn_group
        self.mlp_ratio = mlp_ratio
        self.drop = drop
        self.drop_path = drop_path
        self.ls_init = layer_scale_init_value
        self.rs_init = res_scale_init_value
        self.build()
    
    def computational(self, c) -> nn.Module:
        return DCNFormer(c, self.mlp_ratio, self.dcn_g, self.drop, self.drop_path, self.ls_init, self.rs_init, self.norm, self.act)

class CSP_DCNv4(CSPWrapper):
    def __init__(self, c1, c2, n=2, dcn_g=None, transition1=True, transition2=True, act=nn.GELU, norm=nn.BatchNorm2d, e=0.5) -> None:
        super().__init__(c1, c2, n, transition1, transition2, e, act, norm)
        self.dcn_g = int((c1*e) // 16) if dcn_g is None else dcn_g
        
class DCNv4_Stage(Sequentially):
    def __init__(self, c1, c2, n):
        super().__init__(c1, c2, n)
        self.act = nn.GELU()
        self.build()
        
    def computational(self, c) -> nn.Module:
        return nn.Sequential(
            DeformConv2d_v4(c, c, 3, 1, group=self.dcn_g, dw_kernel_size=3),
            LayerNorm2d(c), # Force DCNv4 to use Layer Normalization
            self.act()
        )
        
class CSP_DCNFormer(CSPWrapper):
    def __init__(self, c1, c2, n=2, dcn_group=None, transition1=True, transition2=True,
                 mlp_ratio: int=4, drop: float=0, drop_path: float=0, 
                 layer_scale_init_value: float=None, 
                 res_scale_init_value: float=None, 
                 act=nn.GELU, norm=LayerNorm2d,
                 e=0.5) -> None:
        super().__init__(c1, c2, n, transition1, transition2, e, act, norm)
        self.dcn_g = dcn_group
        self.mlp_ratio = mlp_ratio
        self.drop = drop
        self.drop_path = drop_path
        self.ls_init = layer_scale_init_value
        self.rs_init = res_scale_init_value
        self.build()
        
    def computational(self, c) -> nn.Module:
        return DCNFormer(c, self.mlp_ratio, self.dcn_g, self.drop, self.drop_path, self.ls_init, self.rs_init, self.norm, self.act)
        
class Stage_PureDCNv4(Sequentially):
    def __init__(self, c1, c2, n, dcn_g=None, act=nn.GELU):
        super().__init__(c1, c2, n)
        self.dcn_g = dcn_g
        self.act = eval(act) if isinstance(act, str) else act
        self.build()
        
    def computational(self, c) -> nn.Module:
        return nn.Sequential(
            DeformConv2d_v4(c, c, 3, 1, group=self.dcn_g, dw_kernel_size=3),
            LayerNorm2d(c),
            self.act()
        )
     
class Stage_DCNFormer(Sequentially):
    def __init__(self, c1, c2, n,
                 dcn_group: int=None, mlp_ratio=4,
                 drop: float = 0, drop_path: float = 0, 
                 layer_scale_init_value: float = None, 
                 res_scale_init_value: float = None, 
                 act: Type[nn.Module] = nn.GELU,
                 norm: Type[nn.Module] = LayerNorm2d, ):
        super().__init__(c1, c2, n)
        self.dcn_g = dcn_group
        self.act = act
        self.norm = norm
        self.mlp_ratio = mlp_ratio
        self.drop = drop
        self.drop_path = drop_path
        self.ls_init = layer_scale_init_value
        self.rs_init = res_scale_init_value
        
        self.build()
    
    def computational(self, c) -> nn.Module:
        return DCNFormer(c, self.mlp_ratio, self.dcn_g, self.drop, self.drop_path, self.ls_init, self.rs_init, self.norm, self.act)
    
class AvgDownsample(nn.Module):
    """ADown with exposed Norm. and Act."""

    def __init__(self, c1, c2, act=nn.GELU, norm=nn.LayerNorm):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = CNA(c1 // 2, self.c, 3, 2, 1, norm=norm, act=act)
        self.cv2 = CNA(c1 // 2, self.c, 1, 1, 0, norm=norm, act=act)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)
    