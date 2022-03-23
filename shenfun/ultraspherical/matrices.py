import functools
import sympy as sp
import numpy as np
from shenfun.matrixbase import SpectralMatrix, extract_diagonal_matrix
from . import bases

Q  = bases.Orthogonal
CD = bases.CompactDirichlet
CN = bases.CompactNeumann
P1 = bases.Phi1
P2 = bases.Phi2
P3 = bases.Phi3
P4 = bases.Phi4


class BQQmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(Q_j, Q_k)_w,

    :math:`Q_k \in` :class:`.ultraspherical.bases.Orthogonal` and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], Q)
        assert isinstance(trial[0], Q)
        from shenfun.jacobi.recursions import h, n, cn, alfa
        N = test[0].N
        k = np.arange(N, dtype=int)
        a = test[0].alpha
        hh = h(a, a, n, 0, cn)
        d = {0: np.zeros(N)}
        d[0][:] = sp.lambdify(n, hh)(k)
        d[0][0] = sp.simplify(h(alfa, alfa, n, 0, cn).subs(n, 0)).subs(alfa, a) # For Chebyshev, need the correct limit
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

class BCDCDmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\phi_j, \phi_k)_w,

    :math:`\phi_k \in` :class:`.ultraspherical.bases.CompactDirichlet` and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], CD)
        assert isinstance(trial[0], CD)
        B = BQQmat((test[0].get_orthogonal(), 0), (trial[0].get_orthogonal(), 0))
        K0 = test[0].stencil_matrix()
        K1 = trial[0].stencil_matrix()
        M = test[0].N
        N = test[0].N
        K0.shape = (M-2, M)
        K1.shape = (N-2, N)
        D = K0.diags('csr')*B.diags('csr')*K1.diags('csr').T
        DD = extract_diagonal_matrix(D, lowerband=2, upperband=2)
        SpectralMatrix.__init__(self, DD._storage, test, trial, scale=scale, measure=measure)

class BCNCNmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\phi_j, \phi_k)_w,

    :math:`\phi_k \in` :class:`.ultraspherical.bases.CompactNeumann` and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], CN)
        assert isinstance(trial[0], CN)
        B = BQQmat((test[0].get_orthogonal(), 0), (trial[0].get_orthogonal(), 0))
        K0 = test[0].stencil_matrix()
        K1 = trial[0].stencil_matrix()
        M = test[0].N
        N = test[0].N
        K0.shape = (M-2, M)
        K1.shape = (N-2, N)
        D = K0.diags('csr')*B.diags('csr')*K1.diags('csr').T
        DD = extract_diagonal_matrix(D, lowerband=2, upperband=2)
        SpectralMatrix.__init__(self, DD._storage, test, trial, scale=scale, measure=measure)


class _Ultramatrix(SpectralMatrix):
    def __init__(self, test, trial, scale=1, measure=1):
        SpectralMatrix.__init__(self, {}, test, trial, scale=scale, measure=measure)


class _UltraMatDict(dict):
    """Dictionary of inner product matrices

    Matrices that are missing keys are generated from Vandermonde type
    computations.

    """

    def __missing__(self, key):
        measure = 1 if len(key) == 2 else key[3]
        c = functools.partial(_Ultramatrix, measure=measure)
        self[key] = c
        return c

    def __getitem__(self, key):
        matrix = dict.__getitem__(self, key)
        return matrix

mat = _UltraMatDict({
    ((Q,  0), (Q,  0)): BQQmat,
    ((CD, 0), (CD, 0)): BCDCDmat,
    ((CN, 0), (CN, 0)): BCNCNmat,
})
