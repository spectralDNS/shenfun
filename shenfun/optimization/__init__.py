"""Module for optimized functions

Some methods performed in Python may be slowing down solvers. In this optimization
module we place optimized functions that are to be used instead of default
Python methods. Some methods are implemented solely in Cython and only called
from within the regular Python modules.

"""
import importlib
from functools import wraps

class runtimeoptimizer:
    """Decorator that chooses optimized function at runtime

    At runtime the decorator looks at

        config['optimization']['mode']
        config['optimization']['verbose']

    and returns the optimized function of choice.
    """
    def __init__(self, func):
        self.func = func
    def __call__(self, *args, **kwargs):
        fun = optimizer(self.func)
        fun(*args, **kwargs)

def optimizer(func, wrap=True):
    """Decorator used to wrap calls to optimized versions of functions.

    Parameters
    ----------
    func : The function to optimize
    wrap : bool, optional
        If True, return function wrapped using functools wraps.
        If False, return unwrapped function.
    """
    from shenfun.config import config
    mod = config['optimization']['mode']
    verbose = config['optimization']['verbose']

    if mod.lower() not in ('cython', 'numba'):
        # Use python function
        if verbose:
            print(func.__qualname__ + ' not optimized')
        return func
    mod = importlib.import_module('shenfun.optimization.'+mod.lower())
    fun = getattr(mod, func.__name__, func)
    if fun is func:
        fun = getattr(mod, func.__qualname__.replace('.', '_'), func)
    if verbose:
        if fun is func:
            print(fun.__qualname__ + ' not optimized')
    if wrap is False:
        return fun
    @wraps(func)
    def wrapped_function(*args, **kwargs):
        u0 = fun(*args, **kwargs)
        return u0
    return wrapped_function
