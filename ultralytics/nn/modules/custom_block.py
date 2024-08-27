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
)
    
class GELAN_SwinV2(nn.Module):
    def __init__(self, c1, c2, n=1, heads=3, window_size=7, g=2, e=0.5, act_csp=nn.GELU, use_checkpoint=False):
        # in_ch (auto), out_ch (arg[0]), repeat (depth), head, window_size, GELAN_width, csp_expandr
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.cv1 = Conv(c1, c2, 1, 1, act=act_csp) # In/Out Channel match
        self.cv2 = Conv((2 + n) * self.c, c2, 1, act=act_csp)  # Final Concat
        self.r = nn.ModuleList(SwinV2Block(self.c, heads, window_size) for _ in range(g))
        self.m = nn.ModuleList(self.r for _ in range(n))
        self.use_checkpoint = use_checkpoint

    def create_mask(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device, dtype=x.dtype)  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        csp, x = self.cv1(x).chunk(2, 1)
        densed = [x, csp]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        attn_mask = self.create_mask(x, H, W)
        for itr1 in self.m:
            for itr2 in itr1:
                itr2.H, itr2.W = H, W
                # if not (torch.jit.is_scripting() and self.use_checkpoint):
                #     x = checkpoint.checkpoint(itr2, x, attn_mask)
                # else:
                x = itr2(x, attn_mask)
            densed.append(x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous())
        return self.cv2(torch.cat(densed, 1))

    def forward_split(self, x):
        csp, x = self.cv1(x).split((self.c, self.c), 1)
        densed = [x, csp]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        attn_mask = self.create_mask(x, H, W)
        for itr1 in self.m:
            for itr2 in itr1:
                itr2.H, itr2.W = H, W
                # if not (torch.jit.is_scripting() and self.use_checkpoint):
                #     x = checkpoint.checkpoint(itr2, x, attn_mask)
                # else:
                x = itr2(x, attn_mask)
            densed.append(x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous())
        return self.cv2(torch.cat(densed, 1))

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        c2 (int): Number of output channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, c2, norm_layer=nn.LayerNorm):
        super().__init__()
        assert c2 == (2 * dim)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

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

        x = self.reduction(x)
        x = self.norm(x)

        x = x.view(B, int(H/2), int(W/2), C*2)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        c1 (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, c1=3, embed_dim=96, patch_size=4, norm_layer=None):
        super().__init__()
        self.patch_size = (patch_size, patch_size)

        self.in_chans = c1
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(c1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        f_pad = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if f_pad:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm(x)
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
