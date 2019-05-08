"""
Module for implementing helper functions.
"""
import types
from collections import MutableMapping
import numpy as np
from scipy.fftpack import dct
from shenfun.optimization import optimizer


__all__ = ['inheritdocstrings', 'clenshaw_curtis1D', 'CachedArrayDict', 'outer']

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

class CachedArrayDict(MutableMapping):
    """Dictionary for caching Numpy arrays (work arrays)

    Example
    -------

    >>> import numpy as np
    >>> from shenfun.utilities import CachedArrayDict
    >>> work = CachedArrayDict()
    >>> a = np.ones((3, 4), dtype=int)
    >>> w = work[(a, 0, True)] # create work array with shape as a
    >>> print(w.shape)
    (3, 4)
    >>> print(w)
    [[0 0 0 0]
     [0 0 0 0]
     [0 0 0 0]]
    >>> w2 = work[(a, 1, True)] # Get different(note 1!) array of same shape/dtype
    """
    def __init__(self):
        self._data = {}

    def __getitem__(self, key):
        newkey, fill = self.__keytransform__(key)
        try:
            value = self._data[newkey]
        except KeyError:
            shape, dtype, _ = newkey
            value = np.zeros(shape, dtype=np.dtype(dtype, align=True))
            self._data[newkey] = value
        if fill:
            value.fill(0)
        return value

    @staticmethod
    def __keytransform__(key):
        assert len(key) == 3
        return (key[0].shape, key[0].dtype, key[1]), key[2]

    def __len__(self):
        return len(self._data)

    def __setitem__(self, key, value):
        self._data[self.__keytransform__(key)[0]] = value

    def __delitem__(self, key):
        del self._data[self.__keytransform__(key)[0]]

    def __iter__(self):
        return iter(self._data)

    def values(self):
        raise TypeError('Cached work arrays not iterable')

def outer(a, b, c):
    r"""Return outer product $c_{i,j} = a_i b_j$

    Parameters
    ----------
    a : Array of shape (N, ...)
    b : Array of shape (N, ...)
    c : Array of shape (N*N, ...)

    The outer product is taken over the first index of a and b,
    for all remaining indices.
    """
    av = a.v
    bv = b.v
    cv = c.v
    symmetric = a is b
    if av.shape[0] == 2:
        outer2D(av, bv, cv, symmetric)
    elif av.shape[0] == 3:
        outer3D(av, bv, cv, symmetric)
    return c

@optimizer
def outer2D(a, b, c, symmetric):
    c[0] = a[0]*b[0]
    c[1] = a[0]*b[1]
    if symmetric:
        c[2] = c[1]
    else:
        c[2] = a[1]*b[0]
    c[3] = a[1]*b[1]

@optimizer
def outer3D(a, b, c, symmetric):
    c[0] = a[0]*b[0]
    c[1] = a[0]*b[1]
    c[2] = a[0]*b[2]
    if symmetric:
        c[3] = c[1]
        c[6] = c[2]
        c[7] = c[5]
    else:
        c[3] = a[1]*b[0]
        c[6] = a[2]*b[0]
        c[7] = a[2]*b[1]
    c[4] = a[1]*b[1]
    c[5] = a[1]*b[2]
    c[8] = a[2]*b[2]
