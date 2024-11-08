from abc import ABC, abstractmethod
import inspect
try:
    from typing import Self, Type  # type: ignore # Python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # Python <3.11
from functools import partial

import torch
import torch.nn as nn

from .ops import *
from .definitions import *

class GELANWrapper(nn.Module, ABC):
    r""" Generalized Efficient Layer Aggeration Network (GELAN) wrapper.
    # Make your network gradient like Academia Sinica!

    Notes:
        * Optional arguments with `None` given are allowed. 
            * If `None` given, the value will be the default. This is conventional with yaml model architecture definition.
        * Developer should implement computational(...) abstract method and then call build() to make the wrapper ready. 
    
    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Depth of the stage (Number of computation groups), Default: 2
        g (int, optional): Number of Computational blocks in group. Default: 1
        transition (bool, optional): Activate "Fusion First" transition layer. Detail refers to [1] Fig. 4c. Default: True
        e (float, optional): CSP Expandsion. Default: 0.5
        act (callable[[], nn.Module]], optional): Activation. Default: nn.GELU
        norm (callable[[int], nn.Module]], optional): Normalization. Default: timm.models.convnext.LayerNorm2d copies
    
    ### References:
        [1] C.-Y. Wang, I-H. Yeh, and H.-Y. M. Liao, "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information," arXiv preprint arXiv:2402.13616, Feb. 21, 2024.
        [2] C.-Y. Wang, H.-Y. M. Liao, and I-H. Yeh, "Designing Network Design Strategies Through Gradient Path Analysis," Journal of Information Science and Engineering, Vol. 39 No. 4, pp. 975-995, 2023.
    """
    def __init__(self, c1: int, c2: int,
                n: Optional[int]=2, g: Optional[int]=1, transition: Optional[bool]=True, e: Optional[float]=0.5, 
                act:  Optional[Callable[[],    nn.Module]]=nn.GELU, 
                norm: Optional[Callable[[int], nn.Module]]=LayerNorm2d,):
        super().__init__()
        # Handle parameters initialization.
        self.c1 = c1
        self.c2 = c2
        self.n = n if n is not None else 2
        self.g = g if g is not None else 1
        self.e = e if e is not None else 0.5
        self.transition = transition if transition is not None else True
        
        # Handle Act and Norm.
        if (isinstance(act, str)):  act  = eval(act)
        if (isinstance(norm, str)): norm = eval(norm)
        if (act  is None): act  = nn.GELU
        if (norm is None): norm = LayerNorm2d
        assert isinstance(norm, type) and isinstance(act, type)
        self.act = act
        self.norm = norm
        
        # Makes GELAN inner-parameter
        self._c = int(c2 * self.e)  
        self.__ready = False
        
    def build(self) -> Self:
        self.cv1 = CNA(self.c1, self.c2, 1, 1, act=self.act, norm=self.norm) 
        self.ct1 = self.transition_layer((1 + self.n) * self._c, (1 + self.n) * self._c, k=1, act=self.act, norm=self.norm) \
                if self.transition else nn.Identity()
        self.ct2 = CNA((2 + self.n) * self._c, self.c2, 1, act=self.act, norm=self.norm)
        # Degrade the GELAN to CSP to prevent repeated CSP gradient.
        if ((self.n == 1) and (self.transition==False)): 
            self.ct2 = CNA(2 * self._c, self.c2, 1, act=self.act, norm=self.norm)
        self.r = nn.ModuleList([self.computational(self._c) for _ in range(self.g*self.n)])
        self.__ready = True
        return self
        
    def conv_norm_act(self, c1, c2, **kwargs) -> nn.Module:
        return CNA(c1, c2, **kwargs)
    
    def transition_layer(self, c1, c2, **kwargs) -> nn.Module:
        return self.conv_norm_act(c1, c2, **kwargs)
    
    @abstractmethod
    def computational(self, c) -> nn.Module:
        raise NO_COMPUTATIONAL_IMPLEMENTED
    
    def forward(self, x):
        '''
        Forward pass of the GELAN wrapper.

        Raises:
            NotImplementedError: If the wrapper has not been built yet.
        '''
        if (not self.__ready): 
            raise NO_BUILD
            
        csp, x = self.cv1(x).chunk(2, 1)
        densed = [x] if not ((self.n == 1) and (self.transition == False)) \
            else []
        
        for idx, itr in enumerate(self.r, start=0):
            x = itr(x)
            if ((idx+1) % self.g == 0):
                densed.append(x)
                
        if (self.transition is False): 
            return self.ct2(torch.cat([csp, *densed], 1))
        return self.ct2(torch.cat([csp, self.ct1(torch.cat(densed, 1))], 1))

class Sequentially(nn.Module, ABC):
    def __init__(self, c1: int, c2: int, n: int):
        ''' Simply wrapper to make block connect together in sequence to make a simple Backbone Stage.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of depths (repeat times of computational block).
        '''
        super().__init__()
        if (c1 != c2):
            raise CHANNEL_MISMATCH_NOTALLOW(c1, c2)
        self.c = c1
        self.n = n
        self.seq = None
        self.__ready = False

    @abstractmethod
    def computational(self, c) -> nn.Module:
        raise NO_COMPUTATIONAL_IMPLEMENTED

    def build(self):
        self.__ready = True
        self.seq = nn.Sequential(*[self.computational(self.c) for _ in range(self.n)])
        return self
    
    def forward(self, x):
        if (not self.__ready): 
            raise NO_BUILD
        assert isinstance(self.seq, nn.Sequential)
        return self.seq(x)

class MetaNeXt(nn.Module, ABC):
    """ MetaNeXt Block wrapper.
    Something advanced computational architecture modified from MetaFormer.
    
    Notes:
        * Optional arguments with `None` given are allowed. 
            * If `None` given, the value will be the default. This is conventional with yaml model architecture definition.
        * Developer should implement token_mixer(...) abstract method and then call build() to make the wrapper ready. 
    Args:
        c (int): Number of input channels.
        mlp_ratio (int, optional): MLP ratio. Default: 4.
        drop_path (float, optional): Drop path probability. Default: 0.
        init_ls (float, optional): Init value for Layer Scale. Default: 1e-6.
        act (callable[[], nn.Module]], optional): Activation. Default: nn.GELU
        norm (callable[[int], nn.Module]], optional): Normalization. Default: timm.models.convnext.LayerNorm2d copies

    References:
        [1] W. Yu, P. Zhou, S. Yan, and X. Wang, "InceptionNeXt: When Inception Meets ConvNeXt," in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle WA, USA, Jun. 17, 2024.
    """
    def __init__(self, 
            c: int,
            mlp_ratio: Optional[int]=4,
            drop_path: Optional[float]=0.,
            init_ls:   Optional[float]=1e-6,
            act:       Optional[Callable[[],    nn.Module]]=nn.GELU, 
            norm:      Optional[Callable[[int], nn.Module]]=LayerNorm2d,):
        super().__init__()
        
        # Handle parameters initialization.
        self.c = c
        self.mlp_ratio  = mlp_ratio if mlp_ratio is not None else 4
        self.drop_path  = drop_path if drop_path is not None else 0.
        self.init_ls    = init_ls   if init_ls   is not None else 1e-6
        
        # Handle Act and Norm.
        if (isinstance(act, str)):  act  = eval(act)
        if (isinstance(norm, str)): norm = eval(norm)
        if (act  is None): act  = nn.GELU
        if (norm is None): norm = LayerNorm2d
        assert isinstance(norm, type) and isinstance(act, type)
        self.act = act
        self.norm = norm
        
        self.__ready = False
        
    @abstractmethod
    def token_mixer(self, c: int) -> nn.Module:
        raise NO_COMPUTATIONAL_IMPLEMENTED
    
    def MLP_layer(self, c) -> nn.Module:
        return ConvMLP(c, c, int(c*self.mlp_ratio), self.act, None)
    
    def build(self) -> Self:
        self.token_mixer = self.token_mixer(self.c)
        self.norm1 = self.norm(self.c)
        self.mlp = self.MLP_layer(self.c)
        self.gamma = nn.Parameter(self.init_ls * torch.ones(self.c)) if self.init_ls else None
        self.drop = DropPath(self.drop_path)
        self.__ready = True
        return self
    
    def forward(self, x):
        if (not self.__ready): 
            raise NO_BUILD
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm1(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop(x) + shortcut
        return x

class MetaFormer(nn.Module, ABC):
    """ MetaFormer Block wrapper

    Args: 
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls (float): Init value for Layer Scale. Default: 1e-6.

    References:
        TODO
    """    

    def __init__(self, 
            c,
            mlp_ratio=4,
            drop: float=0., drop_path: float=0.,
            layer_scale_init_value: float=None,
            res_scale_init_value: float=None,
            norm: Type[nn.Module]=LayerNorm2d,
            act: Type[nn.Module]=nn.GELU,
            token_mixer: Type[nn.Module]=None):
        super().__init__()
        
        if (isinstance(act, str)): act = eval(act)
        if (isinstance(norm, str)): norm = eval(norm)
        self.__ready = False

        self.c = c
        self.mlp_ratio = mlp_ratio
        self.ls_init = layer_scale_init_value
        self.rs_init = res_scale_init_value
        self.drop = drop
        self.drop_path = drop_path
        self.norm = eval(norm) if isinstance(norm, str) else norm 
        self.act  = eval(act)  if isinstance(act,  str) else act  
        self.token_mixer = token_mixer
        
    def token_mixer_layer(self, c: int) -> nn.Module:
        return self.token_mixer(c) if self.token_mixer is not None else nn.Identity()
    
    def MLP_layer(self, c) -> nn.Module:
        return ConvMLP(c, c, int(c*self.mlp_ratio), self.act, None, False, self.drop)
    
    def build(self) -> Self:
        self.token_mixer = self.token_mixer_layer(self.c)
        self.mlp   = self.MLP_layer(self.c)
        self.norm1 = self.norm(self.c)
        self.norm2 = self.norm(self.c)
        self.ls1   = Scale(self.c, self.ls_init) if self.ls_init \
            else nn.Identity()
        self.ls2   = Scale(self.c, self.ls_init) if self.ls_init \
            else nn.Identity()
        self.rs1   = Scale(self.c, self.rs_init) if self.rs_init \
            else nn.Identity()
        self.rs2   = Scale(self.c, self.rs_init) if self.rs_init \
            else nn.Identity()
        self.drop_path1 = DropPath(self.drop_path) if self.drop_path > 0.\
            else nn.Identity()
        self.drop_path2 = DropPath(self.drop_path) if self.drop_path > 0.\
            else nn.Identity()
        self.__ready = True
        return self
    
    def forward(self, x):
        if (not self.__ready): 
            raise NO_BUILD
        x = self.rs1(x) + \
            self.ls1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.rs2(x) + \
            self.ls2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x

    
class CSPWrapper(nn.Module, ABC):
    def __init__(self, c1, c2, n=2, transition1=True, transition2=True, e=0.5, act=nn.GELU, norm=nn.BatchNorm2d) -> None:
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.c = int(c2 * e)  
        self.n = n
        self.transition1 = transition1
        self.transition2 = transition2
        self.e = e
        if (isinstance(act, str)):  act  = eval(act)
        if (isinstance(norm, str)): norm = eval(norm)
        self.act = act
        self.norm = norm
        self.__ready = False
    
    @abstractmethod
    def computational(self, c) -> nn.Module:
        raise NO_COMPUTATIONAL_IMPLEMENTED
    
    def build(self) -> Self:
        self.cv1 = CNA(self.c1, self.c2, 1, 1, act=self.act, norm=self.norm) 
        self.ct1 = CNA(self.c, self.c, 1, act=self.act, norm=self.norm) \
                if self.transition1 else nn.Identity()
        self.ct2 = CNA(2*self.c, self.c2, 1, act=self.act, norm=self.norm) \
                if self.transition2 else nn.Identity()
        self.r = nn.Sequential(*[self.computational(self.c) for _ in range(self.n)])
        self.__ready = True
        return self

    def forward(self, x):
        if (not self.__ready): 
            raise NO_BUILD
        csp, x = self.cv1(x).chunk(2, 1)
        x = self.r(x)
        return self.ct2(torch.cat([csp, self.ct1(x)], 1))