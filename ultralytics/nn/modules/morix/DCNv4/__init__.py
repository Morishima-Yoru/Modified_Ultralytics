try:
    from .functions import DCNv4Function, FlashDeformAttnFunction
    from .modules import DCNv4, FlashDeformAttn
except:
    class DCNv4:
        pass
    class FlashDeformAttn:
        pass
    class DCNv4Function:
        pass
    class FlashDeformAttnFunction:
        pass