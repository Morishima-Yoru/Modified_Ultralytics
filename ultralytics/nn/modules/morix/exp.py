
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
from .ops import DWCNA
from .normalization import *
from .DCNv4 import *
from ....morix.utils.formatting import *

import torch.nn.functional as F
from .definitions import *

class DeformedSubPixelConv(nn.Module):
    def __init__(self, c1: int, c2: int, 
                 upscale_factor: int=2, 
                 kernel_size: int=3,
                 act=nn.GELU, norm=LayerNorm2d):
        super().__init__()
        self.c2 = c2 if c2 > 0 else c1
        self.r = r = upscale_factor
        self._cr =  c2 * (r ** 2)
        if (isinstance(norm, str)): norm = eval(norm)
        if (isinstance(act,  str)): act  = eval(act)
        assert isinstance(norm, type) and isinstance(act, type)
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

