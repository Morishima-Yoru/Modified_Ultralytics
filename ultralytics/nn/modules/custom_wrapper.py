from abc import *
import inspect
try:
    from typing import Self, Type  # Python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # Python <3.11
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .custom_ops import *
    

class GELAN_Wrapper(nn.Module, ABC):
    r""" Generalized Efficient Layer Aggeration Network (GELAN) wrapper
        Make your stages gradient like Academia Sinica!

    Args: TODO
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of Computational blocks, Default: 2
        g (int, optional): Number of groups with Computational blocks (Number of Swin Transformer Block every Comp. blocks contains). Default: 2
        e (float, optional): CSP Expandsion. Default: 0.5
        transition Union[bool, nn.Module]: Cross Staga Partial fusion strategies control. Can specify transition operation with nn.Module type which args wrappered to (in_dim, out_dim). Default: (False, True) a.k.a Fusion First.
        act (nn.Module, optional): Activation for stray convolution. Default: nn.GELU
    References:
        C.-Y. Wang, I-H. Yeh, and H.-Y. M. Liao; "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information":"; arXiv Preprint arXiv:2402.13616, Feb. 2024.
        C.-Y. Wang, H.-Y. M. Liao, and I-H. Yeh; "Designing Network Design Strategies Through Gradient Path Analysis"; Journal of Information Science and Engineering, Vol. 39 No. 4, pp. 975-995.
    """
    def __init__(self, c1, c2, n=2, g=2, transition=True, e=0.5, act=nn.GELU, norm=LayerNorm2d):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.c = int(c2 * e)  
        self.n = n
        self.g = g
        self.transition = transition
        self.e = e
        self.act = act
        self.norm = norm
        self.__ready = False
        
    def build(self) -> Self:
        self.cv1 = CNA(self.c1, self.c2, 1, 1, act=self.act, norm=self.norm) 
        self.ct1 = self.transition_layer((1 + self.n) * self.c, (1 + self.n) * self.c, k=1, act=self.act, norm=self.norm) if self.transition == True else nn.Identity()
        self.ct2 = CNA((2 + self.n) * self.c, self.c2, 1, act=self.act, norm=self.norm)
        self.r = nn.ModuleList(self.computational(self.c) for _ in range(self.g*self.n))
        self.__ready = True
        return self
        
    def conv_norm_act(self, c1, c2, **kwargs) -> nn.Module:
        _kwargs = self.filter_kwargs(CNA, **kwargs)
        return CNA(c1, c2, **_kwargs)
    
    def transition_layer(self, c1, c2, **kwargs) -> nn.Module:
        _kwargs = self.filter_kwargs(self.conv_norm_act, **kwargs)
        return self.conv_norm_act(c1, c2, **_kwargs)
    
    @abstractmethod
    def computational(self, c) -> nn.Module:
        raise NotImplementedError("Attributes not initialized properly")
    
    @staticmethod
    def filter_kwargs(func, **kwargs):
        sig = inspect.signature(func)
        valid_args = sig.parameters.keys()
        return {k: v for k, v in kwargs.items() if k in valid_args}

    def computational_dimchange(self, x, direction: int):
        # For something method which needs dim change before and after computation.
        # In full B, C, H, W dim network. There is no anything needs to do.
        return x
    
    def forward(self, x):
        if (not self.__ready): 
            raise NotImplementedError("Attributes not initialized properly")
            
        csp, x = self.cv1(x).chunk(2, 1)
        densed = [x]
        x = self.computational_dimchange(x, 0)
        for idx, itr in enumerate(self.r, start=1):
            x = itr(x)
            if idx % self.g == 0:
                densed.append(self.computational_dimchange(x, 1))
                
        if (self.transition is False): return self.ct2(torch.cat([csp, densed], 1))
        return self.ct2(torch.cat([csp, 
                    self.ct1(torch.cat(densed, 1))], 1)
                )

class MetaNeXt(nn.Module, ABC):
    """ MetaNeXt Block
    Args: TODO
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, 
            c,
            mlp_ratio=4,
            drop_path: float=0.,
            ls: float=1e-6,
            norm: Type[nn.Module]=LayerNorm2d,
            act: Type[nn.Module]=nn.GELU,
            token_mixer: Type[nn.Module]=None):
        super().__init__()
        self.__ready = False

        self.c = c
        self.mlp_ratio = mlp_ratio
        self.ls = ls
        self.drop_path = drop_path
        self.norm = norm
        self.act = act
        self.token_mixer = token_mixer
        
    def token_mixer_layer(self, c) -> nn.Module:
        return self.token_mixer(c) if self.token_mixer is not None else nn.Identity()
    
    def MLP_layer(self, c) -> nn.Module:
        return ConvMLP(c, c, int(c*self.mlp_ratio), self.act, self.norm)
    
    def build(self) -> Self:
        self.token_mixer = self.token_mixer_layer(self.c)
        self.norm1 = self.norm(self.c)
        self.mlp = self.MLP_layer(self.c)
        self.gamma = nn.Parameter(self.ls * torch.ones(self.c)) if self.ls else None
        self.drop_path = DropPath(self.drop_path) if self.drop_path > 0. else nn.Identity()
        self.__ready = True
        return self
    
    def forward(self, x):
        if (not self.__ready): 
            raise NotImplementedError("Attributes not initialized properly")
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm1(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x