import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from ..block import ResNetBlock
from .custom_ops import *
from .custom_wrapper import *

__all__ = (
    "GELAN_SwinV2",
    "PatchMerging",
    "Patchify",
    "PatchEmbed",
    "GELAN_InceptionNeXt",
    "GELAN_ConvNeXt",
    "ELAN",
    "CNA",
    "GELAN_MetaNeXt_Ident",
    "Seq_Test",
    "ConvNeXtStage",
    "InceptionNeXtStage"
)
    


class GELAN_SwinV2(nn.Module):
    r""" Generalized Efficient Layer Aggeration Network (GELAN) styled Swin Transformer v2 feature extraction stage.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of Computational blocks, Default: 2
        heads (int, optional): Number of Swin Transformer Block heads. Default: 3
        window_sz (int, optional): Number of Swin Transformer Block window size. Default: 7
        g (int, optional): Number of groups with Computational blocks (Number of Swin Transformer Block every Comp. blocks contains). Default: 2
        e (float, optional): CSP Expandsion. Default: 0.5
        transition Union[bool, nn.Module]: Cross Staga Partial fusion strategies control. Can specify transition operation with nn.Module type which args wrappered to (in_dim, out_dim). Default: (False, True) a.k.a Fusion First.
        act (nn.Module, optional): Activation for stray convolution. Default: nn.GELU
    References:
        C.-Y. Wang, I-H. Yeh, and H.-Y. M. Liao; YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information; arXiv Preprint arXiv:2402.13616, Feb. 2024.
        C.-Y. Wang, H.-Y. M. Liao, and I-H. Yeh; Designing Network Design Strategies Through Gradient Path Analysis; Journal of Information Science and Engineering, Vol. 39 No. 4, pp. 975-995.
        Z. Liu, H. Hu, Y. Lin, Z. Yao, Z. Xie, Y. Wei, J. Ning, Y. Cao, Z. Zhang, L. Dong, F. Wei, and B. Guo; Swin Transformer V2: Scaling Up Capacity and Resolution; International Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
    """
    def __init__(self, c1, c2, n=2, heads=3, window_sz=7, g=2, e=0.5, transition=True, act=nn.GELU):
        # in_ch (auto), out_ch (arg[0]), repeat (depth), head, window_size, GELAN_width, csp_expandr
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.g = g # ELAN group
        self.wsz = window_sz
        self.ssz = window_sz // 2
        self.cv1 = CNA(c1, c2, 1, 1, act=act, norm=LayerNorm2d) # In/Out Channel match
        if (isinstance(transition, bool)):
            self.ct1 = CNA((1 + n) * self.c, (1 + n) * self.c, 1, act=act, norm=nn.LayerNorm) if transition == True else nn.Identity()
        else: self.ct1 = transition((1 + n) * self.c, (1 + n) * self.c)
        self.ct2 = CNA((2 + n) * self.c, c2, 1, act=act, norm=LayerNorm2d)

        self.r = nn.ModuleList(SwinV2Block(self.c, heads, window_sz, act_layer=act) for _ in range(g*n))

    def create_mask(self, x, H, W):
        Hp = int(np.ceil(H / self.wsz)) * self.wsz
        Wp = int(np.ceil(W / self.wsz)) * self.wsz
        # calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device, dtype=x.dtype)  # 1 H W 1
        h_slices = (slice(0, -self.wsz),
                    slice(-self.wsz, -self.ssz),
                    slice(-self.ssz, None))
        w_slices = (slice(0, -self.wsz),
                    slice(-self.wsz, -self.ssz),
                    slice(-self.ssz, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                # Guard for None prevent. ONNX static export purpose
                if (h.stop or H) > (h.start or 0) and (w.stop or W) > (w.start or 0):
                    img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.wsz)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.wsz * self.wsz)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        csp, x = self.cv1(x).chunk(2, 1)
        densed = [x]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        attn_mask = self.create_mask(x, H, W)
        elan_idx = 0
        for itr in self.r:
            itr.H, itr.W = H, W
            x = itr(x, attn_mask)
            elan_idx += 1
            if (elan_idx == self.g):
                elan_idx = 0
                densed.append(x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous())
        return self.ct2(torch.cat([csp, self.ct1(torch.cat(densed, 1))], 1))

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
        norm (Type[nn.Module], optional): Normalization layer.  Default: nn.LayerNorm

        [1] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s", in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), New Orleans, LA, USA, pp. 11976-11986, Jun. 21, 2022.
    """

    def __init__(self, c1, emb, patch_sz=2, norm_first=True, norm: Type[nn.Module]=LayerNorm2d):
        super().__init__()
        self.cv1 = nn.Conv2d(c1, emb, patch_sz, patch_sz)
        nch = c1 if norm_first else emb
        self.norm_first = norm_first
        self.norm = norm(nch) if norm is not None else nn.Identity()

    def forward(self, x):
        return self.cv1(self.norm(x)) if self.norm_first else self.norm(self.cv1(x))
    
class ConvNeXt_Block(MetaNeXt):
    def __init__(self, c, drop_path=0., ls=1e-6):
        """
        Args:
            c (int): Number of channels.
            drop_path (float, optional): Drop path probability. Default: 0.
            ls (float, optional): Label smoothing value. Default: 1e-6.
        """
        
        super().__init__(c, 4, drop_path, ls)
        self.build()
    
    def token_mixer_layer(self, c: int) -> nn.Module:
        return nn.Conv2d(c, c, kernel_size=7, padding=3, groups=c)
    
class InceptionNeXt_Block(MetaNeXt):
    def __init__(self, 
            c,
            mlp_ratio=4,
            drop_path: float=0.,
            ls: float=1e-6,):
        """
        Args:
            c (int): Number of channels.
            mlp_ratio (int, optional): MLP ratio. Default: 4.
            drop_path (float, optional): Drop path probability. Default: 0.
            ls (float, optional): Label smoothing value. Default: 1e-6.
        """
        super().__init__(c, mlp_ratio=mlp_ratio, drop_path=drop_path, ls=ls)
        self.build()
    
    def token_mixer_layer(self, c: int) -> nn.Module:
        return InceptionDWConv2d(c)

class GELAN_InceptionNeXt(GELAN_Wrapper):
    def __init__(self, c1, c2, n, g, mlp_ratio, transition=True, e=0.5, act=nn.GELU, norm=LayerNorm2d):
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
            norm (nn.Module, optional): Normalization layer. Default: LayerNorm2d.
        """
        super().__init__(c1, c2, n, g, transition, e, act, norm)
        self.mlp_ratio = mlp_ratio
        self.build()
    
    def computational(self, c) -> nn.Module:
        return InceptionNeXt_Block(c, self.mlp_ratio)

class GELAN_ConvNeXt(GELAN_Wrapper):
    def __init__(self, c1, c2, n, g, transition=True, e=0.5, act=nn.GELU, norm=LayerNorm2d):
        """
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of repetitions.
            g (int): Number of groups.
            transition (bool, optional): Whether to use transition layer. Default: True.
            e (float, optional): Expansion ratio. Default: 0.5.
            act (nn.Module, optional): Activation function. Default: nn.GELU.
            norm (nn.Module, optional): Normalization layer. Default: LayerNorm2d.
        """
        super().__init__(c1, c2, n, g, transition, e, act, norm)
        self.build()
    
    def computational(self, c) -> nn.Module:
        return ConvNeXt_Block(c)
    
class ConvNeXtStage(Sequentially):
    def __init__(self, c1, c2, n):
        """
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of repetitions.
        """
        super().__init__(c1, c2, n)
        self.build()

    def computational(self, c) -> nn.Module:
        return ConvNeXt_Block(c)
    
class InceptionNeXtStage(Sequentially):
    def __init__(self, c1, c2, n, mlp_ratio=4):
        """
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of repetitions.
            mlp_ratio (int, optional): MLP ratio. Defaults to 4.
        """
        super().__init__(c1, c2, n)
        self.mlp_ratio = mlp_ratio
        self.build()

    def computational(self, c) -> nn.Module:
        return InceptionNeXt_Block(c, mlp_ratio=self.mlp_ratio)

class ELAN(GELAN_Wrapper):
    def __init__(self, c1, c2, n, g, transition=True, e=0.5, act=nn.GELU, norm=nn.BatchNorm2d):
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
    
class GELAN_MetaNeXt_Ident(GELAN_Wrapper):
    def __init__(self, c1, c2, n, g, mlp_ratio, transition=True, e=0.5, act=nn.GELU, norm=LayerNorm2d):
        super().__init__(c1, c2, n, g, transition, e, act, norm)
        self.mlp_ratio = mlp_ratio
        self.build()
    
    @staticmethod
    def tk(c) -> nn.Module:
        return ConvNeXt_Block(c)

    def computational(self, c) -> nn.Module:
        return MetaNeXt(
            c,
            self.mlp_ratio,
            norm=self.norm,
            act=self.act,
            token_mixer=self.tk
        ).build()  
    
class Seq_Test(Sequentially):
    def __init__(self, c1, c2, n):
        super().__init__(c1, c2, n)
        self.build()

    def computational(self, c) -> nn.Module:
        return ResNetBlock(c, c, e=1)

    
