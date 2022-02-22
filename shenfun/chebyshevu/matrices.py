r"""
This module contains specific inner product matrices for the different bases in
the Chebyshev family of the second kind.

A naming convention is used for the first capital letter for all matrices.
The first letter refers to type of matrix.

    - Mass matrices start with `B`
    - One derivative start with `C`
    - Two derivatives (Laplace) start with `A`
    - Four derivatives (Biharmonic) start with `S`

A matrix may consist of different types of test and trialfunctions as long as
they are all in the Chebyshev family, either first or second kind.
The next letters in the matrix name uses the short form for all these
different bases according to

T  = Orthogonal
CD = CompactDirichlet
BD = BCDirichlet
BB = BCBiharmonic
P1 = Phi1
P2 = Phi2
T1 = Theta1

So a mass matrix using CompactDirichlet trial and Phi1 test is named
BP1CDmat.

All matrices in this module may be looked up using the 'mat' dictionary,
which takes test and trialfunctions along with the number of derivatives
to be applied to each. As such the mass matrix BTTmat may be looked up
as

>>> from shenfun.chebyshevu.matrices import mat
>>> from shenfun.chebyshevu.bases import Orthogonal as T
>>> B = mat[((T, 0), (T, 0))]

and an instance of the matrix can be created as

>>> B0 = T(10)
>>> BM = B((B0, 0), (B0, 0))
>>> import numpy as np
>>> d = {-2: np.array([-np.pi/2]),
...       0: np.array([ 1.5*np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]),
...       2: np.array([-np.pi/2])}
>>> [np.all(BM[k] == v) for k, v in d.items()]
[True, True, True]

However, this way of creating matrices is not reccommended use. It is far
more elegant to use the TrialFunction/TestFunction interface, and to
generate the matrix as an inner product:

>>> from shenfun import TrialFunction, TestFunction, inner
>>> u = TrialFunction(B0)
>>> v = TestFunction(B0)
>>> BM = inner(u, v)
>>> [np.all(BM[k] == v) for k, v in d.items()]
[True, True, True]

To see that this is in fact the BSDSDmat:

>>> print(BM.__class__)
<class 'shenfun.chebyshev.matrices.BSDSDmat'>

"""
#pylint: disable=bad-continuation, redefined-builtin

from __future__ import division

import functools
import numpy as np
import sympy as sp
import scipy.sparse as scp
from shenfun.optimization import cython, numba, optimizer
from shenfun.matrixbase import SpectralMatrix, SparseMatrix
from shenfun.la import SparseMatrixSolver
from shenfun.la import TDMA as generic_TDMA
from shenfun.la import PDMA as generic_PDMA
from shenfun.la import TwoDMA, FDMA
from . import bases
from shenfun.chebyshev import bases as chebbases

x = sp.symbols('x', real=True)
xp = sp.symbols('x', real=True, positive=True)

# Short names for instances of bases
U = bases.Orthogonal
CD = bases.CompactDirichlet
P1 = bases.Phi1
P2 = bases.Phi2
BCD = bases.BCDirichlet
BCB = bases.BCBiharmonic

SD = chebbases.ShenDirichlet
SN = chebbases.ShenNeumann

def get_B2(N):
    """Get index-shifted B2 matrix for Chebyshev basis of first kind
    """
    ck = np.ones(N, int)
    ck[0] = 2
    k = np.arange(N+2)
    bm2 = ck/(4*k[2:]*(k[2:]-1))
    b0 = -1/(2*k[2:]**2-2)
    bp2 = 1/(4*k[2:]*(k[2:]+1))
    return scp.diags([bm2, b0, bp2], [0, 2, 4], shape=(N, N+2), format='csr')

class BUUmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        B_{kj}=(\phi_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-1 \text{ and } k = 0, 1, ..., N-1

    :math:`\phi_k = U_k` is a Chebyshev basis function of second kind.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], U)
        assert isinstance(trial[0], U)
        SpectralMatrix.__init__(self, {0: np.pi/2}, test, trial, scale=scale, measure=measure)

class BP1SDmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        B_{kj}=(\psi_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-3 \text{ and } k = 0, 1, ..., N-3

    :math:`\phi_k` is a Phi1 basis function (second kind) and :math:`\psi_j`
    is a ShenDirichlet basis function of first kind.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], P1)
        assert isinstance(trial[0], SD)
        N = test[0].N-2
        K = trial[0].stencil_matrix()
        K.shape = (N, N+2)
        K = K.diags('csr')
        B2 = get_B2(N)
        if not test[0].is_scaled:
            k = np.arange(N+2)
            B2 = SparseMatrix({0: (k[:-2]+2)}, (N, N)).diags('csr')*B2
        M = B2 * K.T
        d = {-2: M.diagonal(-2), 0: M.diagonal(0), 2: M.diagonal(2), 4: M.diagonal(4)}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

class AP1SDmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        A_{kj}=(\psi''_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-3 \text{ and } k = 0, 1, ..., N-3

    :math:`\phi_k` is a Phi1 basis function (second kind) and :math:`\psi_j`
    is a ShenDirichlet basis function of first kind.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], P1)
        assert isinstance(trial[0], SD)
        d = {0: -1, 2: 1}
        if not test[0].is_scaled:
            k = np.arange(test[0].N-2)
            d = {0: -(k+2), 2: k[:-2]+2}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

    def get_solver(self):
        return TwoDMA

class BP1SNmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        B_{kj}=(\psi_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-3 \text{ and } k = 0, 1, ..., N-3

    :math:`\phi_k` is a Phi1 basis function (second kind) and :math:`\psi_j`
    is a ShenNeumann basis function of first kind.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], P1)
        assert isinstance(trial[0], SN)
        N = test[0].N-2
        K = trial[0].stencil_matrix()
        K.shape = (N, N+2)
        K = K.diags('csr')
        B2 = get_B2(N)
        if not test[0].is_scaled:
            B2 = SparseMatrix({0: (k[:-2]+2)}, (N, N)).diags('csr')*B2
        M = B2 * K.T
        d = {-2: M.diagonal(-2), 0: M.diagonal(0), 2: M.diagonal(2), 4: M.diagonal(4)}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

class AP1SNmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        A_{kj}=(\psi''_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-3 \text{ and } k = 0, 1, ..., N-3

    :math:`\phi_k` is a Phi1 basis function (second kind) and :math:`\psi_j`
    is a ShenNeumann basis function of first kind.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], P1)
        assert isinstance(trial[0], SN)
        k = np.arange(test[0].N-2)
        d = {0: -(k/(k+2))**2, 2: 1}
        if not test[0].is_scaled:
            d = {0: -k**2/(k+2), 2: k[:-2]+2}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

    def get_solver(self):
        return TwoDMA

class _Chebumatrix(SpectralMatrix):
    def __init__(self, test, trial, scale=1, measure=1):
        SpectralMatrix.__init__(self, {}, test, trial, scale=scale, measure=measure)

class _ChebuMatDict(dict):
    """Dictionary of inner product matrices

    Matrices that are missing keys are generated from Vandermonde type
    computations.

    """

    def __missing__(self, key):
        measure = 1 if len(key) == 2 else key[3]
        c = functools.partial(_Chebumatrix, measure=measure)
        self[key] = c
        return c

    def __getitem__(self, key):
        matrix = dict.__getitem__(self, key)
        #assert key[0][1] == 0, 'Test cannot be differentiated (weighted space)'
        return matrix



# Define dictionary to hold all predefined matrices
# When looked up, missing matrices will be generated automatically
mat = _ChebuMatDict({
    ((U,  0), (U,  0)): BUUmat,
    ((P1, 0), (SD, 0)): BP1SDmat,
    ((P1, 0), (SN, 0)): BP1SNmat,
    ((P1, 0), (SD, 2)): AP1SDmat,
    ((P1, 0), (SN, 2)): AP1SNmat,

})
