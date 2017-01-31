from __future__ import division

__all__ = ['BLLmat', 'BDDmat', 'BBBmat', 'BNNmat',
           'LegendreMatrices']

import numpy as np
from shenfun.matrixbase import ShenMatrix
from shenfun.utilities import inheritdocstrings
from . import bases
from shenfun.la import TDMA, PDMA

# Short names for instances of bases
LB = bases.LegendreBasis()
SD = bases.ShenDirichletBasis()
SB = bases.ShenBiharmonicBasis()
SN = bases.ShenNeumannBasis()


@inheritdocstrings
class BLLmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (L_j, L_k)_w

    where

        j = 0, 1, ..., N and k = 0, 1, ..., N

    and L_k is the Legendre basis function.

    """
    testfunction = (LB, 0)
    trialfunction = (LB, 0)
    def __init__(self, K):
        N = K.shape[0]
        k = K[:N].astype(float)
        d = {0: 2./(2.*k+1)}
        ShenMatrix.__init__(self, d, N, self.testfunction, self.trialfunction)


class BDDmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (psi_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-2

    and psi_k is the Shen Legendre Dirichlet basis function.

    """
    testfunction = (SD, 0)
    trialfunction = (SD, 0)
    def __init__(self, K):
        N = K.shape[0]
        k = K[:N-2].astype(float)
        d = {-2: -2./(2*k[2:] + 1),
              0: 2./(2.*k+1) + 2./(2*k+5)}
        d[2] = d[-2]
        ShenMatrix.__init__(self, d, N, self.testfunction, self.trialfunction)
        self.solver = TDMA(self)


class BNNmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (psi_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-2

    and psi_k is the Shen Legendre Neumann basis function.

    """
    testfunction = (SN, 0)
    trialfunction = (SN, 0)
    def __init__(self, K):
        N = K.shape[0]
        ShenMatrix.__init__(self, {}, N, self.testfunction, self.trialfunction)


class BBBmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (psi_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-2

    and psi_k is the Shen Legendre Biharmonic basis function.

    """
    testfunction = (SB, 0)
    trialfunction = (SB, 0)
    def __init__(self, K):
        N = K.shape[0]
        ShenMatrix.__init__(self, {}, N, self.testfunction, self.trialfunction)
        self.solver = PDMA(self)

class _Legmatrix(ShenMatrix):
    testfunction = None
    trialfunction = None
    def __init__(self, K):
        assert len(K.shape) == 1
        ShenMatrix.__init__(self, {}, K.shape[0], self.testfunction, self.trialfunction)

class _LegMatDict(dict):
    """Dictionary of inner product matrices

    Matrices that are missing keys are generated from Vandermonde type
    computations.

    """

    def __missing__(self, key):
        c = _Legmatrix
        c.testfunction = key[0]
        c.trialfunction = key[1]
        assert c.testfunction[1] == 0, 'Test cannot be differentiated (weighted space)'
        self[key] = c
        return c

    def __getitem__(self, key):
        matrix = dict.__getitem__(self, key)
        matrix.testfunction = key[0]
        matrix.trialfunction = key[1]
        return matrix


LegendreMatrices = _LegMatDict({
    ((LB, 0), (LB, 0)): BLLmat,
    ((SD, 0), (SD, 0)): BDDmat,
    ((SB, 0), (SB, 0)): BBBmat,
    ((SN, 0), (SN, 0)): BNNmat
    })
