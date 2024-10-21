from contextlib import suppress
def __bootstrap__():
    global __bootstrap__, __loader__, __file__
    import sys, pkg_resources, importlib.util
    __file__ = pkg_resources.resource_filename(__name__, 'ext.cp312-win_amd64.pyd')
    __loader__ = None; del __bootstrap__, __loader__
    spec = importlib.util.spec_from_file_location(__name__,__file__)
    if (spec is None): raise ImportError("Could not find module 'ext'")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
with suppress(ImportError):
    __bootstrap__()
