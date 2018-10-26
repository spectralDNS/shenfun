"""
Module for implementing helper functions.
"""
import types
from collections import MutableMapping
import numpy as np
from scipy.fftpack import dct


__all__ = ['inheritdocstrings', 'clenshaw_curtis1D', 'CachedArrayDict']

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
    >>> w = work[(a, 0)] # create work array with shape as a
    >>> print(w.shape)
    (3, 4)
    >>> print(w)
    [[0 0 0 0]
     [0 0 0 0]
     [0 0 0 0]]
    >>> w2 = work[(a, 1)] # Get different(note 1!) array of same shape/dtype
    """
    def __init__(self):
        self._data = {}

    def __getitem__(self, key):
        newkey = self.__keytransform__(key)
        try:
            value = self._data[newkey]
        except KeyError:
            shape, dtype, _ = newkey
            value = np.zeros(shape, dtype=np.dtype(dtype, align=True))
            self._data[newkey] = value
        value.fill(0)
        return value

    @staticmethod
    def __keytransform__(key):
        assert len(key) == 2
        assert isinstance(key[0], np.ndarray)
        shape = key[0].shape
        dtype = key[0].dtype
        i = key[1]
        return (shape, np.dtype(dtype), i)

    def __len__(self):
        return len(self._data)

    def __setitem__(self, key, value):
        self._data[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self._data[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self._data)

    def values(self):
        raise TypeError('Cached work arrays not iterable')
