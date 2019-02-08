import numpy as np
from shenfun.matrixbase import SpectralMatrix
from shenfun.optimization.cython import Matvec
from shenfun.utilities import inheritdocstrings
from shenfun.la import TDMA
from . import bases

HB = bases.Basis

@inheritdocstrings
class BHHmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (H_j, H_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N

    and :math:`H_k` is the Hermite (function) basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], HB)
        assert isinstance(trial[0], HB)
        SpectralMatrix.__init__(self, {0:1}, test, trial)

    def solve(self, b, u=None, axis=0):
        if u is not None:
            u[:] = b
            u /= self.scale
            return u

        else:
            b /= self.scale
            return b

    def matvec(self, v, c, format='python', axis=0):
        c[:] = v
        self.scale_array(c)
        return c

@inheritdocstrings
class AHHmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        A_{kj} = (H'_j, H'_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N

    and :math:`H_k` is the Hermite (function) basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], HB)
        assert isinstance(trial[0], HB)
        N = test[0].N
        k = np.arange(N, dtype=np.float)
        d = {0: k+0.5,
             2: -np.sqrt((k[:-2]+1)*(k[:-2]+2))/2}
        d[0][-1] = (N-1)/2.
        d[-2] = d[2]
        SpectralMatrix.__init__(self, d, test, trial)
        self.solve = TDMA(self)

    def matvec(self, v, c, format='cython', axis=0):
        N, M = self.shape
        c.fill(0)
        if format == 'cython' and v.ndim == 3:
            ld = self[-2]*np.ones(M-2)
            Matvec.Tridiagonal_matvec3D_ptr(v, c, ld, self[0], ld, axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 2:
            ld = self[-2]*np.ones(M-2)
            Matvec.Tridiagonal_matvec2D_ptr(v, c, ld, self[0], ld, axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 1:
            ld = self[-2]*np.ones(M-2)
            Matvec.Tridiagonal_matvec(v, c, ld, self[0], ld)
            self.scale_array(c)
        elif format == 'self':
            if axis > 0:
                v = np.moveaxis(v, axis, 0)
                c = np.moveaxis(c, axis, 0)
            s = (slice(None),)+(np.newaxis,)*(v.ndim-1) # broadcasting
            c[:(N-2)] = self[2][s]*v[2:N]
            c[:N] += self[0][s]*v[:N]
            c[2:N] += self[-2][s]*v[:(N-2)]
            if axis > 0:
                v = np.moveaxis(v, 0, axis)
                c = np.moveaxis(c, 0, axis)
            self.scale_array(c)

        else:
            c = super(AHHmat, self).matvec(v, c, format=format, axis=axis)

        return c

@inheritdocstrings
class _Lagmatrix(SpectralMatrix):
    def __init__(self, test, trial):
        SpectralMatrix.__init__(self, {}, test, trial)


class _LagMatDict(dict):
    """Dictionary of inner product matrices

    Matrices that are missing keys are generated from Vandermonde type
    computations.

    """

    def __missing__(self, key):
        c = _Lagmatrix
        self[key] = c
        return c

    def __getitem__(self, key):
        matrix = dict.__getitem__(self, key)
        return matrix


mat = _LagMatDict({
    ((HB, 0), (HB, 0)): BHHmat,
    ((HB, 1), (HB, 1)): AHHmat
    })
