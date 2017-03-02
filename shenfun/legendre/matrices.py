from __future__ import division

__all__ = ['mat']

import numpy as np
from shenfun.matrixbase import ShenMatrix
from shenfun.utilities import inheritdocstrings
from . import bases
from shenfun.la import TDMA, PDMA

# Short names for instances of bases
LB = bases.Basis
SD = bases.ShenDirichletBasis
SB = bases.ShenBiharmonicBasis
SN = bases.ShenNeumannBasis


@inheritdocstrings
class BLLmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (L_j, L_k)_w

    where

        j = 0, 1, ..., N and k = 0, 1, ..., N

    and L_k is the Legendre basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], LB)
        assert isinstance(trial[0], LB)
        N = test[0].N
        k = np.arange(N, dtype=np.float)
        d = {0: 2./(2.*k+1)}
        ShenMatrix.__init__(self, d, test, trial)


@inheritdocstrings
class BDDmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (psi_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-2

    and psi_k is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N-2, dtype=np.float)
        d = {-2: -2./(2*k[2:] + 1),
              0: 2./(2.*k+1) + 2./(2*k+5)}
        d[2] = d[-2]
        ShenMatrix.__init__(self, d, test, trial)
        self.solver = TDMA(self)


@inheritdocstrings
class BNNmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (psi_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-2

    and psi_k is the Shen Legendre Neumann basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SN)
        N = test[0].N
        ShenMatrix.__init__(self, {}, test, trial)


@inheritdocstrings
class BBBmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (psi_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-2

    and psi_k is the Shen Legendre Biharmonic basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        N = test[0].N
        ShenMatrix.__init__(self, {}, test, trial)
        self.solver = PDMA(self)


@inheritdocstrings
class ADDmat(ShenMatrix):
    """Stiffness matrix for inner product A_{kj} = (psi'_j, psi'_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-2

    and psi_k is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N-2, dtype=np.float)
        d = {0: 4*k+6}
        ShenMatrix.__init__(self, d, test, trial)

    def solve(self, b, u=None, axis=0):
        N = self.shape[0] + 2
        assert N == b.shape[0]
        s = self.trialfunction[0].slice()

        if u is None:
            u = b
        else:
            assert u.shape == b.shape

        # Move axis to first
        if axis > 0:
            u = np.moveaxis(u, axis, 0)
            if not u is b:
                b = np.moveaxis(b, axis, 0)

        bs = b[s]
        us = u[s]
        d = 1./self[0]
        sl = [np.newaxis]*bs.ndim
        sl[0] = slice(None)
        us[:] = bs*d[sl]
        u[-2] = self.testfunction[0].bc[0]
        u[-1] = self.testfunction[0].bc[1]

        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if not u is b:
                b = np.moveaxis(b, axis, 0)

        return u


@inheritdocstrings
class _Legmatrix(ShenMatrix):
    def __init__(self, test, trial):
        ShenMatrix.__init__(self, {}, test, trial)


class _LegMatDict(dict):
    """Dictionary of inner product matrices

    Matrices that are missing keys are generated from Vandermonde type
    computations.

    """

    def __missing__(self, key):
        c = _Legmatrix
        self[key] = c
        return c

    def __getitem__(self, key):
        matrix = dict.__getitem__(self, key)
        return matrix


mat = _LegMatDict({
    ((LB, 0), (LB, 0)): BLLmat,
    ((SD, 0), (SD, 0)): BDDmat,
    ((SB, 0), (SB, 0)): BBBmat,
    ((SN, 0), (SN, 0)): BNNmat,
    ((SD, 1), (SD, 1)): ADDmat,
    })
