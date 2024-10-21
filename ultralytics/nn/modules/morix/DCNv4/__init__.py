import warnings


try:
    from .functions import DCNv4Function, FlashDeformAttnFunction # type: ignore
    from .modules import DCNv4, FlashDeformAttn                   # type: ignore
except ImportError:
    warnings.warn(
        "Failed to import DCNv4 and FlashDeformAttn modules. "

        "DCNv4 and FlashDeformAttn are not available.", UserWarning
    )

    class DCNv4:
        pass

    class FlashDeformAttn:
        pass

    class DCNv4Function:
        pass

    class FlashDeformAttnFunction:
        pass
