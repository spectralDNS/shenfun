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

    - T  = Orthogonal
    - CD = CompactDirichlet
    - CN = CompactNeumann
    - BCG = BCGeneric
    - P1 = Phi1
    - P2 = Phi2
    - P3 = Phi3
    - P4 = Phi4

So a mass matrix using CompactDirichlet trial and test is named
BCDCDmat.

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
>>> d = {0: np.pi/2}
>>> [np.all(BM[k] == v) for k, v in d.items()]
[True]

However, this way of creating matrices is not reccommended use. It is far
more elegant to use the TrialFunction/TestFunction interface, and to
generate the matrix as an inner product:

>>> from shenfun import TrialFunction, TestFunction, inner
>>> u = TrialFunction(B0)
>>> v = TestFunction(B0)
>>> BM = inner(u, v)
>>> [np.all(BM[k] == v) for k, v in d.items()]
[True]

To see that this is in fact the BUUmat:

>>> print(BM.__class__)
<class 'shenfun.chebyshevu.matrices.BUUmat'>

"""
#pylint: disable=bad-continuation, redefined-builtin

from __future__ import division

import functools
import numpy as np
import sympy as sp
from shenfun.matrixbase import SpectralMatrix, SparseMatrix
from shenfun.la import TwoDMA
from shenfun.chebyshev import bases as chebbases
from . import bases

x = sp.symbols('x', real=True)
xp = sp.symbols('x', real=True, positive=True)

# Short names for instances of bases
U = bases.Orthogonal
CD = bases.CompactDirichlet
CN = bases.CompactNeumann
UD = bases.UpperDirichlet
LD = bases.LowerDirichlet
CB = bases.CompositeBase
P1 = bases.Phi1
P2 = bases.Phi2
P3 = bases.Phi3
P4 = bases.Phi4
BCG = bases.BCGeneric

SD = chebbases.ShenDirichlet
SN = chebbases.ShenNeumann

def get_UU(M, N, quad):
    """Return main diagonal of :math:`(U_i, U_j)_w`

    Parameters
    ----------
    M : int
        The number of quadrature points in the test function
    N : int
        The number of quadrature points in the trial function
    quad : str
        Type of quadrature

        - GC - Chebyshev-Gauss
        - GU - Chebyshevu-Gauss
    """
    ll = np.pi/2
    if quad == 'GC' and N >= M:
        ll = np.ones(min(M, N), dtype=float)*ll
        ll[-1] = 2*ll[-1]
    return ll

class BUUmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(U_j, U_k)_w,

    :math:`U_k \in` :class:`.chebyshevu.bases.Orthogonal` and test and trial spaces have
    dimensions of M and N, respectively.

    """
    def assemble(self):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], U)
        assert isinstance(trial[0], U)
        d = get_UU(test[0].N, trial[0].N, test[0].quad)
        return {0: d}

class BP1SDmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshevu.bases.Phi1`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self):
        test, trial = self.testfunction, self.trialfunction
        from shenfun.jacobi.recursions import Lmat, half, cn
        assert isinstance(test[0], P1)
        assert isinstance(trial[0], SD)
        assert test[0].quad == 'GU'
        N = test[0].N-2
        K = trial[0].stencil_matrix()
        K.shape = (N, N+2)
        B2 = Lmat(2, 0, 2, N, N+2, -half, -half, cn) # B^{(2)_{(2)}}
        if not test[0].is_scaled():
            k = np.arange(N+2)
            B2 = SparseMatrix({0: (k[:-2]+2)}, (N, N)).diags('csr')*B2
        M = B2 * K.diags('csr').T
        K.shape = (N+2, N+2)
        d = {-2: M.diagonal(-2), 0: M.diagonal(0), 2: M.diagonal(2), 4: M.diagonal(4)}
        return d

class AP1SDmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj}=(\psi''_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshevu.bases.Phi1`, the trial
    function :math:`\psi_j \in` :class:`.chebyshev.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], P1)
        assert isinstance(trial[0], SD)
        d = {0: -1, 2: 1}
        if not test[0].is_scaled():
            k = np.arange(test[0].N-2)
            d = {0: -(k+2), 2: k[:-2]+2}
        return d

    def get_solver(self):
        return TwoDMA

class BP1SNmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshevu.bases.Phi1`, the
    trial function :math:`\psi_j \in` :class:`.chebyshev.bases.ShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self):
        from shenfun.jacobi.recursions import Lmat, half, cn
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], P1)
        assert isinstance(trial[0], SN)
        N = test[0].N-2
        K = trial[0].stencil_matrix()
        K.shape = (N, N+2)
        B2 = Lmat(2, 0, 2, N, N+2, -half, -half, cn) # B^{(2)_{(2)}}
        if not test[0].is_scaled():
            k = np.arange(test[0].N)
            B2 = SparseMatrix({0: (k[:-2]+2)}, (N, N)).diags('csr')*B2
        M = B2 * K.diags('csr').T
        K.shape = (N+2, N+2)
        d = {-2: M.diagonal(-2), 0: M.diagonal(0), 2: M.diagonal(2), 4: M.diagonal(4)}
        return d

class AP1SNmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj}=(\psi''_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshevu.bases.Phi1`, the trial
    function :math:`\psi_j \in` :class:`.chebyshev.bases.ShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], P1)
        assert isinstance(trial[0], SN)
        k = np.arange(test[0].N-2)
        d = {0: -(k/(k+2))**2, 2: 1}
        if not test[0].is_scaled():
            d = {0: -k**2/(k+2), 2: k[:-2]+2}
        return d

    def get_solver(self):
        return TwoDMA

class _Chebumatrix(SpectralMatrix):
    def __init__(self, test, trial, scale=1, measure=1, assemble=None):
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble)

class _ChebuMatDict(dict):
    """Dictionary of inner product matrices

    Matrices that are missing keys are generated from Vandermonde type
    computations.

    """

    def __missing__(self, key):
        measure = 1 if len(key) == 2 else key[2]
        c = functools.partial(_Chebumatrix, measure=measure)
        self[key] = c
        return c

    def __getitem__(self, key):
        if len(key) == 3:
            matrix = functools.partial(dict.__getitem__(self, key),
                                       measure=key[2])
        else:
            matrix = dict.__getitem__(self, key)
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
