"""
Module for implementing helper functions.
"""
import types
import numpy as np
from scipy.fftpack import dct

__all__ = ['inheritdocstrings', 'clenshaw_curtis1D']

def inheritdocstrings(cls):
    """Method used for inheriting docstrings from parent class

    Use as decorator::

         @inheritdocstrings
         class Child(Parent):

    and Child will use the same docstrings as parent even if
    a method is overloaded. The Child class may overload the
    docstring as well and a new docstring defined for a method
    in Child will overload the Parent.
    """
    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType) and not func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
    return cls

def clenshaw_curtis1D(u, quad="GC"):  # pragma: no cover
    """Clenshaw-Curtis integration in 1D"""
    assert u.ndim == 1
    N = u.shape[0]
    if quad == 'GL':
        w = np.arange(0, N, 1, dtype=float)
        w[2:] = 2./(1-w[2:]**2)
        w[0] = 1
        w[1::2] = 0
        ak = dct(u, 1)
        ak /= (N-1)
        return np.sqrt(np.sum(ak*w))

    assert quad == 'GC'
    d = np.zeros(N)
    k = 2*(1 + np.arange((N-1)//2))
    d[::2] = (2./N)/np.hstack((1., 1.-k*k))
    w = dct(d, type=3)
    return np.sqrt(np.sum(u*w))
