"""
Quasi-inverse matrices for Tau and Galerkin methods

@article{julien09,
title = {Efficient multi-dimensional solution of PDEs using Chebyshev spectral methods},
journal = {Journal of Computational Physics},
volume = {228},
number = {5},
pages = {1480-1503},
year = {2009},
issn = {0021-9991},
doi = {https://doi.org/10.1016/j.jcp.2008.10.043}
}
"""
import numpy as np
from shenfun.matrixbase import SparseMatrix

__all__ = ('QIGmat', 'QITmat', 'QICGmat', 'QICTmat')

# Note to self. Matrices below do not use ck because we use them in
# scalar products not inverted with (T_j, T_k)_w = ck*pi/2 \delta_{kj}

class QImat(SparseMatrix):

    def __mul__(self, y):
        """Returns copy of self.__mul__(y) <==> self*y"""
        if isinstance(y, SparseMatrix):
            return y.__quasi__(self)
        return SparseMatrix.__mul__(self, y)

class QIGmat(QImat):
    """Quasi-inverse matrix for the Galerkin method

    Parameters
    ----------
    N : int
        The number of quadrature points

    """
    def __init__(self, N):
        k = np.arange(N)
        #d = {
        #    0: 1/4/(k[2:]*(k[2:]-1)),
        #    2: -1/2/(k[2:-2]**2-1),
        #    4: 1/4/(k[2:-4]*(k[2:-4]+1))}
        #SparseMatrix.__init__(self, d, (N-2, N-2))
        d = {
            0: 1/4/(k[2:]*(k[2:]-1)),
            2: -1/2/(k[2:]**2-1),
            4: 1/4/(k[2:-2]*(k[2:-2]+1))}
        # Note: truncating the upper diagonals is in agreement with
        # \cite{julien09}.
        d[2][-2:] = 0
        d[4][-2:] = 0
        SparseMatrix.__init__(self, d, (N-2, N))

class QITmat(QImat):
    """Quasi-inverse matrix for the Tau method

    Parameters
    ----------
    N : int
        The number of quadrature points

    """
    def __init__(self, N):
        k = np.arange(N)
        d = {
            -2: 1/4/(k[2:]*(k[2:]-1)),
            0: np.zeros(N),
            2: np.zeros(N-2)}
        d[0][2:-2] = -1/2/(k[2:-2]**2-1)
        d[2][2:-2] = 1/4/(k[2:-4]*(k[2:-4]+1))

        # Truncate, like \cite{julien09}
        d[0][-2:] = 0
        d[2][-2:] = 0
        SparseMatrix.__init__(self, d, (N, N))

class QICGmat(QImat):
    """Quasi-inverse matrix for the Galerkin method

    Parameters
    ----------
    N : int
        The number of quadrature points

    """
    def __init__(self, N):
        k = np.arange(N)
        d = {
            0: 1/(2*(k[:-1]+1)),
            2: -1/(2*(k[:-2]+1))}
        SparseMatrix.__init__(self, d, (N-1, N))

class QICTmat(QImat):
    """Quasi-inverse matrix for the Tau method.

    Parameters
    ----------
    N : int
        The number of quadrature points

    """
    def __init__(self, N):
        k = np.arange(N)
        kk = np.arange(N)
        kk[0] = 1
        dk = np.ones(N)
        dk[0] = 0
        d = {
            -1: 1/(2*(k[:-1]+1)),
             1: -dk[:-1]/(2*(kk[:-1]))}
        SparseMatrix.__init__(self, d, (N, N))
