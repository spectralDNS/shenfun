import functools
import sympy as sp
import numpy as np
from shenfun.matrixbase import SpectralMatrix
from . import bases

J  = bases.Orthogonal
CD = bases.CompactDirichlet
CN = bases.CompactNeumann
P1 = bases.Phi1
P2 = bases.Phi2
P3 = bases.Phi3
P4 = bases.Phi4


class BJJmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(J_j, J_k)_w,

    :math:`J_k \in` :class:`.jacobi.bases.Orthogonal` and test and trial spaces have
    dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], J)
        assert isinstance(trial[0], J)
        from .recursions import h, n
        N = test[0].N
        k = np.arange(N, dtype=int)
        a = test[0].alpha
        b = test[0].beta
        hh = h(a, b, n, 0)
        d = {0: sp.lambdify(n, hh)(k)}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

    def solve(self, b, u=None, axis=0, constraints=()):
        N = self.shape[0]
        assert N == b.shape[axis]
        s = self.trialfunction[0].slice()

        if u is None:
            u = b
        else:
            assert u.shape == b.shape

        # Move axis to first
        if axis > 0:
            u = np.moveaxis(u, axis, 0)
            if u is not b:
                b = np.moveaxis(b, axis, 0)

        bs = b[s]
        us = u[s]
        d = 1./self[0]
        sl = [np.newaxis]*bs.ndim
        sl[0] = slice(None)
        us[:] = bs*d[tuple(sl)]
        u /= self.scale

        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if u is not b:
                b = np.moveaxis(b, axis, 0)

        return u


class _Jacmatrix(SpectralMatrix):
    def __init__(self, test, trial, scale=1, measure=1):
        SpectralMatrix.__init__(self, {}, test, trial, scale=scale, measure=measure)


class _JacMatDict(dict):
    """Dictionary of inner product matrices

    Matrices that are missing keys are generated from Vandermonde type
    computations.

    """

    def __missing__(self, key):
        measure = 1 if len(key) == 2 else key[3]
        c = functools.partial(_Jacmatrix, measure=measure)
        self[key] = c
        return c

    def __getitem__(self, key):
        matrix = dict.__getitem__(self, key)
        return matrix

mat = _JacMatDict({
    ((J,  0), (J,  0)): BJJmat,
})
