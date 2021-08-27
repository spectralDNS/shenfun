"""Module for optimized functions

Some methods performed in Python may be slowing down solvers. In this optimization
module we place optimized functions that are to be used instead of default
Python methods. Some methods are implemented solely in Cython and only called
from within the regular Python modules.

"""
import os
import importlib
from functools import wraps
from . import cython
try:
    from . import numba
except ModuleNotFoundError:
    numba = None

def optimizer(func):
    """Decorator used to wrap calls to optimized versions of functions."""
    from shenfun.config import config
    mod = config['optimization']['mode']
    verbose = config['optimization']['verbose']
    if mod.lower() not in ('cython', 'numba'):
        # Use python function
        if verbose:
            print(func.__name__ + ' not optimized')
        return func
    mod = importlib.import_module('shenfun.optimization.'+mod.lower())
    fun = getattr(mod, func.__name__, func)
    if verbose:
        if fun is func:
            print(fun.__name__ + ' not optimized')

    @wraps(func)
    def wrapped_function(*args, **kwargs):
        u0 = fun(*args, **kwargs)
        return u0

    return wrapped_function
