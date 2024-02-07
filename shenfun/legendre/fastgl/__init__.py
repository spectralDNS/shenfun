try:
    from . import fastgl_wrap
except ModuleNotFoundError:
    print('fastgl not found')
    fastgl_wrap = None

leggauss = getattr(fastgl_wrap, 'leggauss', None)
getGLPair = getattr(fastgl_wrap, 'getGLPair', None)