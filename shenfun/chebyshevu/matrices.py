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
from shenfun.matrixbase import SpectralMatrix, SparseMatrix, extract_diagonal_matrix
from shenfun.la import TwoDMA
from shenfun.utilities import split
from shenfun import config
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

class BUUmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(U_j, U_k)_w,

    :math:`U_k \in` :class:`.chebyshevu.bases.Orthogonal` and test and trial spaces have
    dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], U)
        assert isinstance(trial[0], U)
        SpectralMatrix.__init__(self, {0: np.pi/2}, test, trial, scale=scale, measure=measure)

class BP1SDmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshevu.bases.Phi1`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        from shenfun.jacobi.recursions import Lmat, half, cn
        assert isinstance(test[0], P1)
        assert isinstance(trial[0], SD)
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
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

class AP1SDmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj}=(\psi''_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshevu.bases.Phi1`, the trial
    function :math:`\psi_j \in` :class:`.chebyshev.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], P1)
        assert isinstance(trial[0], SD)
        d = {0: -1, 2: 1}
        if not test[0].is_scaled():
            k = np.arange(test[0].N-2)
            d = {0: -(k+2), 2: k[:-2]+2}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

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
    def __init__(self, test, trial, scale=1, measure=1):
        from shenfun.jacobi.recursions import Lmat, half, cn
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
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

class AP1SNmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj}=(\psi''_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshevu.bases.Phi1`, the trial
    function :math:`\psi_j \in` :class:`.chebyshev.bases.ShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], P1)
        assert isinstance(trial[0], SN)
        k = np.arange(test[0].N-2)
        d = {0: -(k/(k+2))**2, 2: 1}
        if not test[0].is_scaled():
            d = {0: -k**2/(k+2), 2: k[:-2]+2}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

    def get_solver(self):
        return TwoDMA

class BGBCGmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, \phi_k)_w,

    where the test function :math:`\phi_k` is a subclass of
    :class:`.chebyshevu.bases.CompositeBase`, the
    trial :math:`\psi_j \in` :class:`.chebyshevu.bases.BCGeneric`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], CB)
        assert isinstance(trial[0], BCG)
        B = BUUmat((test[0].get_orthogonal(domain=(-1, 1)), 0), (trial[0].get_orthogonal(domain=(-1, 1)), 0))
        K = test[0].stencil_matrix()
        K.shape = (test[0].dim(), test[0].N)
        S = extract_diagonal_matrix(trial[0].stencil_matrix().T).diags('csr')
        q = sp.degree(measure)
        if measure != 1:
            from shenfun.jacobi.recursions import pmat, half, un, a
            assert sp.sympify(measure).is_polynomial()

            A = sp.S(0)
            for dv in split(measure, expand=True):
                sc = dv['coeff']
                msi = dv['x']
                qi = sp.degree(msi)
                Ax = pmat(a, qi, half, half, test[0].N, test[0].N, un)
                A = A + sc*Ax.diags('csr')
            A = K.diags('csr') * A.T * B.diags('csr') * S
        else:
            A = K.diags('csr') * B.diags('csr') * S

        M = B.shape[1]
        K.shape = (test[0].N, test[0].N)
        d = extract_diagonal_matrix(A, lowerband=M+q, upperband=M)
        SpectralMatrix.__init__(self, dict(d), test, trial, scale=scale, measure=measure)

class BGGmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, x^q \phi_k)_w,

    where the test and trial functions :math:`\phi_k` and :math:`\psi_j` are
    any subclasses of :class:`.chebyshevu.bases.CompositeBase` and :math:`q \ge 0`
    is an integer. Test and trial spaces have dimensions of M and N, respectively.

    Note
    ----
    Creating mass matrices this way is efficient in terms of memory since the
    mass matrix of the orthogonal basis is diagonal.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], CB)
        assert isinstance(trial[0], CB)
        B = BUUmat((test[0].get_orthogonal(domain=(-1, 1)), 0), (trial[0].get_orthogonal(domain=(-1, 1)), 0))
        K = test[0].stencil_matrix()
        K.shape = (test[0].dim(), test[0].N)
        S = trial[0].stencil_matrix()
        S.shape = (trial[0].dim(), trial[0].N)
        q = sp.degree(measure)
        if measure != 1:
            from shenfun.jacobi.recursions import pmat, half, un, a
            assert sp.sympify(measure).is_polynomial()
            A = sp.S(0)
            for dv in split(measure, expand=True):
                sc = dv['coeff']
                msi = dv['x']
                qi = sp.degree(msi)
                Ax = pmat(a, qi, half, half, test[0].N, test[0].N, un)
                A = A + sc*Ax.diags('csr')
            A = K.diags('csr') * B.diags('csr') * A * S.diags('csr').T

        else:
            A = K.diags('csr') * B.diags('csr') * S.diags('csr').T

        K.shape = (test[0].N, test[0].N)
        S.shape = (trial[0].N, trial[0].N)
        ub = test[0].N-test[0].dim()+q
        lb = trial[0].N-trial[0].dim()+q
        d = extract_diagonal_matrix(A, lowerband=lb, upperband=ub)
        SpectralMatrix.__init__(self, dict(d), test, trial, scale=scale, measure=measure)

class PXGmat(SpectralMatrix):
    r"""Matrix :math:`D=(d_{ij}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        d_{ij}=(\partial^{k-l} \psi_j, x^q \phi_i)_w,

    where the test function :math:`\phi_i` is in one of :class:`.chebyshevu.bases.Phi1`,
    :class:`.chebyshevu.bases.Phi2`, :class:`.chebyshevu.bases.Phi3`, :class:`.chebyshevu.bases.Phi4`,
    the trial :math:`\psi_j` any class in :class:`.chebyshevu.bases`,
    The three parameters k, q and l are integers, and test and trial spaces
    have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], (P1, P2, P3, P4))
        assert test[0].quad == 'GC'
        from shenfun.jacobi.recursions import Lmat, half, un
        q = sp.degree(measure)
        k = (test[0].N-test[0].dim())//2
        l = k-trial[1]
        if q > 0 and test[0].domain != test[0].reference_domain():
            D = sp.S(0)
            for dv in split(measure, expand=True):
                sc = dv['coeff']
                msi = dv['x']
                qi = sp.degree(msi)
                Ax = Lmat(k, qi, l, test[0].dim(), trial[0].N, half, half, un)
                D = D + sc*Ax
        else:
            D = Lmat(k, q, l, test[0].dim(), trial[0].N, half, half, un)

        if trial[0].is_orthogonal:
            D = extract_diagonal_matrix(D, lowerband=q-k+l, upperband=q+k+l)
        else:
            K = trial[0].stencil_matrix()
            K.shape = (trial[0].dim(), trial[0].N)
            keys = np.sort(np.array(list(K.keys())))
            lb, ub = -keys[0], keys[-1]
            D = extract_diagonal_matrix(D*K.diags('csr').T, lowerband=q-k+l+ub, upperband=q+k+l+lb)
            K.shape = (trial[0].N, trial[0].N)
        SpectralMatrix.__init__(self, dict(D), test, trial, scale=scale, measure=measure)

class PXBCGmat(SpectralMatrix):
    r"""Matrix :math:`D=(d_{ij}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        d_{ij}=(\partial^{k-l}\psi_j, x^q \phi_i)_w,

    where the test function :math:`\phi_i` is in one of :class:`.chebyshevu.bases.Phi1`,
    :class:`.chebyshevu.bases.Phi2`, :class:`.chebyshevu.bases.Phi3`, :class:`.chebyshevu.bases.Phi4`,
    trial :math:`\psi_j \in` :class:`.chebyshevu.bases.BCGeneric`.
    The three parameters k, q, l are integers and test and trial spaces have
    dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], (P1, P2, P3, P4))
        assert isinstance(trial[0], BCG)
        from shenfun.jacobi.recursions import Lmat, half, un
        M = test[0].dim()
        N = trial[0].dim_ortho
        q = sp.degree(measure)
        k = (test[0].N-test[0].dim())//2
        l = k-trial[1]
        if q > 0 and test[0].domain != test[0].reference_domain():
            D = sp.S(0)
            for dv in split(measure, expand=True):
                sc = dv['coeff']
                msi = dv['x']
                qi = sp.degree(msi)
                Ax = Lmat(k, qi, l, M, N, half, half, un)
                D = D + sc*Ax
        else:
            D = Lmat(k, q, l, M, N, half, half, un)

        K = trial[0].stencil_matrix()
        D = extract_diagonal_matrix(D*extract_diagonal_matrix(K).diags('csr').T, lowerband=N+q, upperband=N)
        SpectralMatrix.__init__(self, dict(D), test, trial, scale=scale, measure=measure)


class _Chebumatrix(SpectralMatrix):
    def __init__(self, test, trial, scale=1, measure=1):
        SpectralMatrix.__init__(self, {}, test, trial, scale=scale, measure=measure)

class _ChebuMatDict(dict):
    """Dictionary of inner product matrices

    Matrices that are missing keys are generated from Vandermonde type
    computations.

    """

    def __missing__(self, key):
        measure = 1 if len(key) == 2 else key[2]
        if key[0][1]+key[1][1] == 0 and sp.sympify(measure).is_polynomial():
            if key[1][0] == BCG:
                c = functools.partial(BGBCGmat, measure=measure)
            else:
                c = functools.partial(BGGmat, measure=measure)
        else:
            c = functools.partial(_Chebumatrix, measure=measure)
        self[key] = c
        return c

    def __getitem__(self, key):
        measure = 1 if len(key) == 2 else key[2]
        if key[0][0] in (P1, P2, P3, P4):
            if key[1][0] == BCG:
                k = ('PX', 1)
            else:
                k = ('PX', 0)
            if key[1][1] > int(key[0][0].short_name()[1]) or key[1][0] in (P1, P2, P3, P4):
                # If the number of derivatives is larger than 1 for P1, 2 for P2 etc,
                # then we need to use quadrature. But it should not be larger if you
                # have designed the scheme appropriately, so perhaps we should throw
                # a warning
                k = key
            matrix = functools.partial(dict.__getitem__(self, k),
                                       measure=measure)
        elif len(key) == 3:
            matrix = functools.partial(dict.__getitem__(self, key),
                                       measure=key[2])
        else:
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
    ('PX', 0): PXGmat,
    ('PX', 1): PXBCGmat,

})
