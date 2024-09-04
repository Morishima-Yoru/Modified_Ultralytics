from enum import Enum

import torch
import torch.nn as nn

from torch import Tensor

class e_Format(Enum):
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
    def to_enum(cls, value: 'e_Format | int | str') -> 'e_Format':
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        if isinstance(value, str):
            return cls.from_string(value)
        raise TypeError(f"Cannot convert {value} to e_Format")

def format_convert(x: torch.Tensor, 
                   fmt_from: e_Format | int | str, 
                   fmt_to:   e_Format | int | str,
                   HW_shape: tuple=None,) -> torch.Tensor:
    
    fmt_from = e_Format.to_enum(fmt_from)
    fmt_to   = e_Format.to_enum(fmt_to)
    if (fmt_from is fmt_to): return x
        
    x_o = x.shape
        
    if ((fmt_from is e_Format.BLC) and
        (HW_shape is None) and
        (x.shape[1] % 2 != 0)):
        raise ValueError("Unable to make L to square H and W. Check your padding or pass original Height and Width through HW_shape.\n \
            Given: {}".format(x.shape))
        
    if ((fmt_from is e_Format.BLC) and
        HW_shape[0] * HW_shape[1] != x.shape[1]):
        raise ValueError("Passed HW_shape mismatch with L dim. \n \
            Given: {}, {} ({}), Except: {}"\
            .format(tH, tW, tH*tW,
                    x.shape[1]))
    
    tH, tW = HW_shape if (HW_shape is not None) else (x.shape[2] // 2, x.shape[2] // 2) #TODO
    
    _MEMO = {
        e_Format.BCHW: {
            e_Format.BCHW: None,
            e_Format.BHWC: lambda x: x.permute(0, 2, 3, 1),
            e_Format.BLC : lambda x: x.permute(0, 2, 3, 1).contiguous().flatten(1, 2),
        },
        e_Format.BHWC: {
            e_Format.BCHW: lambda x: x.permute(0, 3, 1, 2),
            e_Format.BHWC: None,
            e_Format.BLC : lambda x: x.flatten(1, 2),
        },
        e_Format.BLC: {
            e_Format.BCHW: lambda x: x.permute(0, 2, 1).contiguous().view(x_o[0], x_o[2], tH, tW),
            e_Format.BHWC: lambda x: x.view(x_o[0], tH, tW, x_o[2]),
            e_Format.BLC : None,
        }
    }
    
    convert = _MEMO.get(fmt_from, {}).get(fmt_to, None)
    if (convert is None):
        raise ValueError(f"Conversion from {fmt_from} to {fmt_to} is not supported.")
        
    return convert(x.contiguous())

def get_HW_shape(x: torch.Tensor, fmt: e_Format | int | str) -> tuple:
    fmt = e_Format.to_enum(fmt)
    if (fmt is e_Format.BLC):
        raise ValueError(f"{fmt} doesn't have Height and Width")
    _MEMO = {
        e_Format.BCHW: lambda x: (x.size(2), x.size(3)),
        e_Format.BHWC: lambda x: (x.size(1), x.size(2)),
    }
    return _MEMO.get(fmt)(x)