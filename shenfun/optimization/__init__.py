"""Module for optimized functions

Some methods performed in Python may be slowing down solvers. In this optimization
module we place optimized functions that are to be used instead of default
Python methods. Some methods are implemented solely in Cython and only called
from within the regular Python modules.

"""
import os
import importlib
from functools import wraps

def optimizer(func):
    """Decorator used to wrap calls to optimized versions of functions."""

    mod = os.environ.get('SHENFUN_OPTIMIZATION', 'cython')
    if mod.lower() not in ('cython', 'numba'):
        # Use python function
        #print(func.__name__ + ' not optimized')
        return func
    mod = importlib.import_module('shenfun.optimization.'+mod.lower())
    fun = getattr(mod, func.__name__, func)
    #if fun is func:
    #    print(fun.__name__ + ' not optimized')

    @wraps(func)
    def wrapped_function(*args, **kwargs):
        u0 = fun(*args, **kwargs)
        return u0

    return wrapped_function
