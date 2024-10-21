from enum import Enum
from typing import Optional, Tuple, Union

import torch

from torch import Tensor

class ShapeFormat(Enum):
    BCHW = 1
    BHWC = 2
    BLC  = 3
    
    @classmethod
    def from_string(cls, fmt_str: str):
        # Define a mapping from string representations to enum values
        mapping = {
            'BCHW': cls.BCHW,
            'NCHW': cls.BCHW,
            'BHWC': cls.BHWC,
            'NHWC': cls.BHWC,
            'BLC' : cls.BLC,
        }
        # Look up the format_str in the mapping and return the corresponding enum value
        if fmt_str in mapping:
            return mapping[fmt_str]
        else:
            raise ValueError(f"Unknown format string: {fmt_str}")
        
    @classmethod
    def to_enum(cls, value: Union['ShapeFormat', int, str]) -> 'ShapeFormat':
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        if isinstance(value, str):
            return cls.from_string(value)
        raise TypeError(f"Cannot convert {value} to e_Format")

class ShapeFormatHelper():
    __FN = {
        ShapeFormat.BCHW: {
            ShapeFormat.BCHW: lambda x, *_: x,
            ShapeFormat.BHWC: lambda x, *_: x.permute(0, 2, 3, 1),
            ShapeFormat.BLC : lambda x, *_: x.permute(0, 2, 3, 1).contiguous().flatten(1, 2),
        },
        ShapeFormat.BHWC: {
            ShapeFormat.BCHW: lambda x, *_: x.permute(0, 3, 1, 2),
            ShapeFormat.BHWC: lambda x, *_: x,
            ShapeFormat.BLC : lambda x, *_: x.flatten(1, 2),
        },
        ShapeFormat.BLC: {
            ShapeFormat.BCHW: lambda x, tH, tW: x.permute(0, 2, 1).contiguous().view(-1, x.shape[-1], tH, tW),
            ShapeFormat.BHWC: lambda x, tH, tW: x.view(-1, tH, tW, x.shape[-1]),
            ShapeFormat.BLC : lambda x, *_: x,
        }
    }
    def __init__(self, x: Tensor, format: Union[ShapeFormat, str]) -> None:
        self.x = x
        self.fmt = format
        
    def to_format(self, format: Union[ShapeFormat, str]) -> Tensor:
        return self.format_convert(self.x, self.fmt, format)
    @classmethod
    def format_convert(cls, x: Tensor, 
                    fmt_from: Union[ShapeFormat, str], 
                    fmt_to:   Union[ShapeFormat, str],
                    HW_shape: Optional[Tuple[int, int]]=None,) -> Tensor:
        
        fmt_from = ShapeFormat.to_enum(fmt_from)
        fmt_to   = ShapeFormat.to_enum(fmt_to)
        if (fmt_from is fmt_to): return x
            
        # _x_shp = x.shape
            
        # if (fmt_from is ShapeFormat.BLC):
            
        #     (HW_shape is None) and
        #     (x.shape[1] % 2 != 0)):
        #     raise ValueError("Unable to make L to square H and W. Check your padding or pass original Height and Width through HW_shape.\n \
        #         Given: {}".format(x.shape))
            
        # if ((fmt_from is ShapeFormat.BLC) and
        #     HW_shape[0] * HW_shape[1] != x.shape[1]):
        #     raise ValueError("Passed HW_shape mismatch with L dim. \n \
        #         Given: {}, {} ({}), Except: {}"\
        #         .format(tH, tW, tH*tW,
        #                 x.shape[1]))
        if (fmt_from is ShapeFormat.BLC):
            tH, tW = HW_shape if (HW_shape is not None) else (x.shape[2] // 2, x.shape[2] // 2) #TODO
        else: tH, tW = (None, None)
        
        convert = cls.__FN.get(fmt_from, {}).get(fmt_to, None)
        if (convert is None):
            raise ValueError(f"Conversion from {fmt_from} to {fmt_to} is not supported.")
            
        return convert(x, tH, tW).contiguous()

    @staticmethod
    def get_HW_shape(x: torch.Tensor, fmt: Union[ShapeFormat, str]) -> tuple:
        fmt = ShapeFormat.to_enum(fmt)
        if (fmt is ShapeFormat.BLC):
            raise ValueError(f"{fmt} doesn't have Height and Width")
        _MEMO = {
            ShapeFormat.BCHW: lambda x: (x.size(2), x.size(3)),
            ShapeFormat.BHWC: lambda x: (x.size(1), x.size(2)),
        }
        return _MEMO.get(fmt)(x)