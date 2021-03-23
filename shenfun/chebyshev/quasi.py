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

__all__ = ('QIGmat', 'QITmat')

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
        d = {
            0: 1/4/(k[2:]*(k[2:]-1)),
            2: -1/2/(k[2:-2]**2-1),
            4: 1/4/(k[2:-4]*(k[2:-4]+1))}
        SparseMatrix.__init__(self, d, (N-2, N-2))

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
        SparseMatrix.__init__(self, d, (N, N))
