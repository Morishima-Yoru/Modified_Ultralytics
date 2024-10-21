try:
    from .flash_deform_attn_func import FlashDeformAttnFunction
    from .dcnv4_func import DCNv4Function
except:
    class FlashDeformAttnFunction:
        pass
    class DCNv4Function:
        pass