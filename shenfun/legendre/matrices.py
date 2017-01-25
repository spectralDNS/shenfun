import numpy as np
from shenfun.matrixbase import ShenMatrix
from shenfun import inheritdocstrings
from . import bases

class BLLmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (L_j, L_k)_w

    where

        j = 0, 1, ..., N and k = 0, 1, ..., N

    and L_k is the Legendre basis function.

    """
    def __init__(self, K, quad='GC'):
        N = K.shape[0]
        k = K[:N].astype(float)
        d = {0: 2./(2.*k+1)}
        trial = bases.LegendreTransform()
        ShenMatrix.__init__(self, d, N, (trial, 0), (trial, 0))

class BDDmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (psi_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-2

    and psi_k is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, K, quad='GC'):
        N = K.shape[0]
        k = K[:N-2].astype(float)
        d = {-2: -2./(2*k[2:] + 1),
              0: 2./(2.*k+1) + 2./(2*k+5)}
        d[2] = d[-2]
        trial = bases.ShenDirichletBasis()
        ShenMatrix.__init__(self, d, N, (trial, 0), (trial, 0))
