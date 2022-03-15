import functools
from shenfun.matrixbase import SpectralMatrix
from shenfun.la import TDMA_O
from . import bases

CD = bases.CompactDirichlet
CN = bases.CompactNeumann
L = bases.Orthogonal


class BLLmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(La_j, La_k)_w,

    :math:`La_k \in` :class:`.laguerre.bases.Orthogonal` and test and trial spaces have
    dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], L)
        assert isinstance(trial[0], L)
        SpectralMatrix.__init__(self, {0:1}, test, trial, scale=scale, measure=measure)

    def solve(self, b, u=None, axis=0, constraints=()):
        if u is not None:
            u[:] = b
            u /= (self.scale*self[0])
            return u

        else:
            b /= (self.scale*self[0])
            return b

    def matvec(self, v, c, format=None, axis=0):
        c[:] = v
        self.scale_array(c, self.scale*self[0])
        return c


class BCDCDmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\phi_j, \phi_k)_w,

    :math:`\phi_k \in` :class:`.laguerre.bases.CompactDirichlet` and test and trial spaces have
    dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], CD)
        assert isinstance(trial[0], CD)
        d = {0:2., 1: -1., -1:-1.}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

    def get_solver(self):
        return TDMA_O


class ACDCDmat(SpectralMatrix):
    r"""Stiffness matrix :math:`BA=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj}=(\phi'_j, \phi'_k)_w,

    :math:`\phi_k \in` :class:`.laguerre.bases.CompactDirichlet` and test and trial spaces have
    dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], CD)
        assert isinstance(trial[0], CD)
        d = {0: 0.5,
             1: 0.25,
             -1: 0.25}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

    def get_solver(self):
        return TDMA_O

class _Lagmatrix(SpectralMatrix):
    def __init__(self, test, trial, measure=1):
        SpectralMatrix.__init__(self, {}, test, trial, measure=measure)


class _LagMatDict(dict):
    """Dictionary of inner product matrices

    Matrices that are missing keys are generated from Vandermonde type
    computations.

    """

    def __missing__(self, key):
        measure = 1 if len(key) == 2 else key[3]
        c = functools.partial(_Lagmatrix, measure=measure)
        self[key] = c
        return c

    def __getitem__(self, key):
        matrix = dict.__getitem__(self, key)
        return matrix


mat = _LagMatDict({
    #((CD, 0), (CD, 0)): BCDCDmat,
    #((CD, 1), (CD, 1)): ACDCDmat,
    ((L, 0), (L, 0)): BLLmat
    })
