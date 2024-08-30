import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .conv import Conv 
from .custom_ops import *

__all__ = (
    "GELAN_SwinV2",
    "PatchMerging",
    "PatchEmbed",
    "CNA",
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
        self.cv1 = CNA(c1, c2, 1, 1, act=act, norm=nn.LayerNorm) # In/Out Channel match
        if (isinstance(transition, bool)):
            self.ct1 = CNA((1 + n) * self.c, (1 + n) * self.c, 1, act=act, norm=nn.LayerNorm) if transition == True else nn.Identity()
        else: self.ct1 = transition((1 + n) * self.c, (1 + n) * self.c)
        self.ct2 = CNA((2 + n) * self.c, c2, 1, act=act, norm=nn.LayerNorm)

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

class Patchify(nn.Module):
    r""" Patch Merging Layer (Called Patchify Stem in ConvNeXt) with full Convolutional network style.
    
    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels. It should be 2 times c1. Otherwise AssertionException.
        patch_sz (int, optional): Number of patch size. Default: 4
        norm (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

        Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie; A ConvNet for the 2020s; IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 11976-11986

    """

    def __init__(self, c1, c2, patch_sz=4, norm=nn.LayerNorm):
        super().__init__()
        assert c2 == (2 * c1)
        self.c1 = c1
        self.cv1 = CNA(c1, c2, patch_sz, patch_sz, act=None, norm=norm)

    def forward(self, x):
        return self.cv1(x)



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


class CNA(nn.Module):
    """Classic Convolution-Normalization-Activation topology"""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=nn.GELU, norm=nn.BatchNorm2d):
        super().__init__()
        norm_equip_gn = {
            # Group Normalization will degrade to Layer Normalization when Groups = 1 (All Channels join to norm).
            # W. Yuxin, and K. He. “Group Normalization.” International Journal of Computer Vision, 2020, Vol. 128, pp. 742 - 755.
            nn.LayerNorm: nn.GroupNorm(1, c2), 
        }
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.norm = norm(c2) if norm not in norm_equip_gn else norm_equip_gn.get(norm)
        self.act = act if isinstance(act, nn.Module) else nn.Identity()

        
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.norm(self.conv(x)))
    
    
    def forward_fuse(self, x):
        # TODO: Not implement.
        return self.act(self.norm(self.conv(x)))