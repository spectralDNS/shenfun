#cython: boundscheck=False
#cython: language_level=3
cimport numpy as np
import numpy as np
cimport cython
import cython
from libc.stdlib cimport malloc, free
from copy import copy
from numpy.polynomial import chebyshev as n_cheb
from .transforms import FMMdirect1, FMMcheb, FMMdirect2, FMMdirect3, \
    FMMdirect4, _leg2cheb, _cheb2leg, _Lambda
#from .transforms cimport get_number_of_blocks

np.import_array()

cdef int get_number_of_blocks(int level, long* D, int N):
    cdef:
        int s = 1
        int nd = 1
        int i, j
    if level > 0:
        for i in range(level):
            nd = 1
            for j in range(i+1):
                nd *= D[level-j-1]
            s += nd
    return s

cdef class FMMLevel:
    cdef public int N
    cdef public int Nn
    cdef public int diags
    cdef public int L
    cdef public int s
    cdef public int maxs
    cdef public int use_direct
    cdef public np.ndarray D
    cdef public np.ndarray Mmin
    cdef public np.ndarray fk
    cdef public np.ndarray Nk
    cdef public np.ndarray T
    cdef public np.ndarray Th
    cdef public np.ndarray ThT

    def __init__(self, int N, int diagonals=8, domains=None, levels=None,
                 int l2c=1, int maxs=100, int use_direct=-1):
        cdef:
            int i, s, rest, h, Nd, level
            np.ndarray[long, ndim=1] doms

        from shenfun.legendre.dlt import getChebyshev, conversionmatrix
        self.N = N
        self.diags = diagonals
        self.maxs = maxs
        self.use_direct = use_direct
        if self.N <= self.use_direct:
            self.Nn = N
            return

        if domains is not None:
            if isinstance(domains, int):
                if levels is None:
                    doms = np.cumsum(domains**np.arange(16))
                    levels = np.where((N//2-diagonals)/doms < maxs)[0][0]
                self.D = np.full(levels, domains, dtype=int)

            else:
                domains = np.atleast_1d(domains)
                levels = len(domains)
                if domains[levels-1] == 1:
                    D0 = domains.copy()
                    for i in range(1000):
                        Nd = get_number_of_blocks(levels, <long*>np.PyArray_DATA(domains), len(domains))
                        s = (N//2-diagonals)//Nd
                        if s < maxs:
                            break
                        else:
                            domains = D0*(i+2)
                self.D = domains
        else:
            if levels is None:
                domains = max(2, int(np.log10(N)))
                doms = np.cumsum(domains**np.arange(16))
                levels = np.where((N//2-diagonals)/doms < maxs)[0][0]
                levels = max(1, levels)
            else:
                for i in range(2, 1000):
                    Nd = np.cumsum(i**np.arange(levels+1))[levels]
                    s = (N//2-diagonals)//Nd
                    if s < maxs:
                        domains = i
                        break
            self.D = np.full(levels, domains, dtype=int)
        self.L = max(1, levels)
        Nd = get_number_of_blocks(self.L, <long*>np.PyArray_DATA(self.D), len(self.D))
        # Create new N with minimal size according to diags and N
        # Pad input array with zeros
        s, rest = divmod(N//2-diagonals, Nd)
        Nn = N
        if rest > 0 or Nn%2 == 1:
            s += 1
            Nn = 2*(diagonals+s*Nd)

        self.Nn = Nn
        self.s = s
        fk = []
        Nk = []
        self.Mmin = np.zeros(self.L, dtype=int)
        for level in range(self.L-1, -1, -1):
            Nx = getChebyshev(level, self.D, s, diagonals, fk, Nk,  l2c)
            self.Mmin[level] = Nx
        self.fk = np.hstack(fk)
        self.Nk = np.array(Nk, dtype=int)
        Mmax = self.Mmin.max()
        TT = {}
        for level in range(self.L-1, -1, -1):
            if self.D[level] not in TT:
                TT[self.D[level]] = conversionmatrix(self.D[level], Mmax)
        self.Th = np.concatenate([TT[d] for d in np.unique(self.D)])
        self.ThT = np.array([copy(self.Th[i].transpose()) for i in range(self.Th.shape[0])])
        self.T = np.zeros((2, s, self.Mmin[-1]))
        Ti = n_cheb.chebvander(np.linspace(-1, 1, 2*s+1)[:-1], self.Mmin[-1]-1)
        self.T[0] = Ti[::2]
        self.T[1] = Ti[1::2]

cdef class Leg2Cheb(FMMLevel):
    cdef public np.ndarray _output_array
    cdef np.ndarray _a
    cdef int axis
    cdef int Mmax
    cdef tuple si

    def __cinit__(self, np.ndarray input_array, output_array=None, int axis=0,
                  int diagonals=8, domains=None, levels=None, int maxs=100, int use_direct=-1):
        self._output_array = input_array.copy() if output_array is None else output_array

    def __init__(self, np.ndarray input_array, output_array=None, int axis=0,
                 int diagonals=8, domains=None, levels=None, int maxs=100, int use_direct=-1):
        cdef:
            int level
            list si = [slice(None)]*input_array.ndim

        FMMLevel.__init__(self, input_array.shape[axis], diagonals=diagonals,
                          domains=domains, levels=levels, maxs=maxs, use_direct=use_direct)
        self._a = _Lambda(np.arange(self.Nn, dtype='d'))
        self.axis = axis
        si[axis] = 0
        self.si = tuple(si)

    def __call__(self, np.ndarray input_array, output_array=None, int transpose=0):

        if input_array.shape[self.axis] <= self.use_direct and input_array.ndim == 1:
            self._output_array[...] = 0
            _leg2cheb(input_array, self._output_array, self._a, transpose)
            if output_array is not None:
                output_array[...] = self._output_array
                return output_array
            return self._output_array

        if transpose == 1:
            input_array = input_array.copy() # preserve input_array
            input_array *= (2/np.pi)
            input_array[self.si] *= 0.5

        if input_array.shape[self.axis] <= self.use_direct:
            self._output_array[...] = 0
            FMMdirect1(input_array, self._output_array, self.axis, self._a, self.N, transpose)
        else:
            self.apply(input_array, self._output_array, transpose)

        if transpose == 0:
            self._output_array *= (2/np.pi)
            self._output_array[self.si] *= 0.5

        if output_array is not None:
            output_array[...] = self._output_array
            return output_array
        return self._output_array

    cdef apply(self, np.ndarray input_array, np.ndarray output_array, int transpose):
        FMMcheb(input_array, output_array, self.axis, self.Nn, self.fk, self.Nk, self.T, self.Th, self.ThT, self.D, self.Mmin, self.s, self.diags, transpose)
        FMMdirect2(input_array, output_array, self.axis, self._a, 2*self.s, get_number_of_blocks(self.L, <long*>np.PyArray_DATA(self.D), len(self.D)), 2*self.diags, transpose)
        FMMdirect1(input_array, output_array, self.axis, self._a, self.diags, transpose)


cdef class Cheb2Leg(FMMLevel):
    cdef public np.ndarray _output_array
    cdef np.ndarray _a
    cdef np.ndarray _dn
    cdef int axis
    cdef tuple s0
    cdef tuple sn1

    def __cinit__(self, np.ndarray input_array, output_array=None, int axis=0,
                  int diagonals=8, domains=None, levels=None, int maxs=100, int use_direct=-1):
        self._output_array = input_array.copy() if output_array is None else output_array

    def __init__(self, np.ndarray input_array, output_array=None, int axis=0,
                 int diagonals=8, domains=None, levels=None, int maxs=100, int use_direct=-1):
        cdef:
            list si = [None]*input_array.ndim
            list sn1 = [slice(None)]*input_array.ndim
            np.ndarray[double] k

        FMMLevel.__init__(self, input_array.shape[axis], diagonals=diagonals,
                          domains=domains, levels=levels, l2c=0, maxs=maxs, use_direct=use_direct)

        k = np.arange(self.Nn, dtype='d')
        k[0] = 1
        self._dn = _Lambda((k[::2]-2)/2)/k[::2]
        self._a = 1/(2*_Lambda(k)*k*(k+0.5))
        self._a[0] = 2/np.sqrt(np.pi)
        self.axis = axis

        # precompute some multidimensional slices
        sn1[axis] = slice(1, self.N)
        self.sn1 = tuple(sn1)
        si[axis] = slice(None)
        self.s0 = tuple(si)

    def __call__(self, np.ndarray input_array, output_array=None):
        cdef:
            np.ndarray w0

        if input_array.shape[self.axis] <= self.use_direct and input_array.ndim == 1:
            self._output_array[...] = 0
            _cheb2leg(input_array, self._output_array, self._dn, self._a)
            if output_array is not None:
                output_array[...] = self._output_array
                return output_array
            return self._output_array
        w0 = input_array.copy()
        w0[self.sn1] *= np.arange(1, self.N)[self.s0]
        if input_array.shape[self.axis] <= self.use_direct:
            self._output_array[...] = 0
            FMMdirect4(w0, self._output_array, self.axis, self._dn[:self.N//2], self._a[:self.N], self.N//2)
        else:
            self.apply(w0, self._output_array)
        self._output_array *= (np.arange(self.N)+0.5)[self.s0]
        if output_array is not None:
            output_array[...] = self._output_array
            return output_array
        return self._output_array

    def apply(self, np.ndarray input_array, np.ndarray output_array):
        FMMcheb(input_array, output_array, self.axis, self.Nn, self.fk, self.Nk, self.T, self.Th, self.ThT, self.D, self.Mmin, self.s, self.diags, False)
        FMMdirect3(input_array, output_array, self.axis, self._dn, self._a, 2*self.s, get_number_of_blocks(self.L, <long*>np.PyArray_DATA(self.D), len(self.D)), 2*self.diags)
        FMMdirect4(input_array, output_array, self.axis, self._dn, self._a, self.diags)
