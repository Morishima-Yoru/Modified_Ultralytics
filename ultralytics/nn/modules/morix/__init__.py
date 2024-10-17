from .ops import (
    CNA,
)

from .custom_head import (
    ClassificationHead
)

from .custom_block import (
    GELAN_SwinV2,
    PatchMerging,
    PatchEmbed,
    Patchify,
    ConvNeXt_Block,
    InceptionNeXt_Block,
    GELAN_InceptionNeXt,
    GELAN_ConvNeXt,
    ConvNeXtStage,
    InceptionNeXtStage,
    ELAN,
    ELAN_DarknetBottleneck,
    DCNFormer,
    GELAN_DCNv4,
    GELAN_DCNFormer,
    CSP_DCNv4,
    DCNv4_Stage,
    CSP_DCNFormer,
    Stage_PureDCNv4,
    Stage_DCNFormer,
    AvgDownsample,
    SubPixelConv,
)

from .normalization import (
    SwitchNorm2d,
    LayerNorm2d,
    SwitchNorm2dNoBatch,
)

PARSE_REQUIRED = (
    CNA,
    GELAN_SwinV2,
    PatchMerging,
    PatchEmbed,
    Patchify,
    ConvNeXt_Block,
    InceptionNeXt_Block,
    GELAN_InceptionNeXt,
    GELAN_ConvNeXt,
    ConvNeXtStage,
    InceptionNeXtStage,
    ELAN,
    ELAN_DarknetBottleneck,
    DCNFormer,
    GELAN_DCNv4,
    GELAN_DCNFormer,
    CSP_DCNv4,
    DCNv4_Stage,
    CSP_DCNFormer,
    Stage_PureDCNv4,
    Stage_DCNFormer,
    AvgDownsample,
    SubPixelConv,
    ClassificationHead,
)
DEPTH_REQUIRED = (
    GELAN_SwinV2,
    GELAN_InceptionNeXt,
    GELAN_ConvNeXt,
    ConvNeXtStage,
    InceptionNeXtStage,
    ELAN,
    ELAN_DarknetBottleneck,
    GELAN_DCNv4,
    GELAN_DCNFormer,
    CSP_DCNv4,
    DCNv4_Stage,
    CSP_DCNFormer,
    Stage_PureDCNv4,
    Stage_DCNFormer,
)

__all__ = (
    'PARSE_REQUIRED',
    'DEPTH_REQUIRED',
    'CNA',
    'GELAN_SwinV2',
    'PatchMerging',
    'PatchEmbed',
    'Patchify',
    'ConvNeXt_Block',
    'InceptionNeXt_Block',
    'GELAN_InceptionNeXt',
    'GELAN_ConvNeXt',
    'ConvNeXtStage',
    'InceptionNeXtStage',
    'ELAN',
    'ELAN_DarknetBottleneck',
    'DCNFormer',
    'GELAN_DCNv4',
    'GELAN_DCNFormer',
    'CSP_DCNv4',
    'DCNv4_Stage',
    'CSP_DCNFormer',
    'Stage_PureDCNv4',
    'Stage_DCNFormer',
    'AvgDownsample',
    'SubPixelConv',
    'SwitchNorm2d',
    'LayerNorm2d',
    'SwitchNorm2dNoBatch',
    'ClassificationHead'
)

