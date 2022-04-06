r"""
This module contains specific inner product matrices for the different bases in
the Chebyshev family.

A naming convention is used for the first capital letter for all matrices.
The first letter refers to type of matrix.

    - Mass matrices start with `B`
    - One derivative start with `C`
    - Two derivatives (Laplace) start with `A`
    - Four derivatives (Biharmonic) start with `S`

A matrix may consist of different types of test and trialfunctions. The next
letters in the matrix name uses the short form for all these different bases
according to

    - T  = Orthogonal
    - SD = ShenDirichlet
    - HH = Heinrichs
    - SB = ShenBiharmonic
    - SN = ShenNeumann
    - CN = CombinedShenNeumann
    - MN = MikNeumann
    - UD = UpperDirichlet
    - LD = LowerDirichlet
    - DN = DirichletNeumann
    - P1 = Phi1
    - P2 = Phi2
    - P3 = Phi3
    - P4 = Phi4
    - BD = BCDirichlet
    - BB = BCBiharmonic

So a mass matrix using ShenDirichlet test and ShenNeumann trial is named
BSDSNmat.

All matrices in this module may be looked up using the 'mat' dictionary,
which takes test and trialfunctions along with the number of derivatives
to be applied to each. As such the mass matrix BSDSDmat may be looked up
as

>>> from shenfun.chebyshev.matrices import mat
>>> from shenfun.chebyshev.bases import ShenDirichlet as SD
>>> B = mat[((SD, 0), (SD, 0))]

and an instance of the matrix can be created as

>>> B0 = SD(10)
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
from shenfun.optimization import cython, numba
from shenfun.matrixbase import SpectralMatrix, SparseMatrix, extract_diagonal_matrix
from shenfun.la import TDMA as generic_TDMA
from shenfun.la import PDMA as generic_PDMA
from shenfun.la import TwoDMA, FDMA
from shenfun.utilities import split
from shenfun import config
from .la import ADDSolver, ANNSolver
from . import bases

x = sp.symbols('x', real=True)
xp = sp.symbols('x', real=True, positive=True)

# Short names for instances of bases
T = bases.Orthogonal
SD = bases.ShenDirichlet
HH = bases.Heinrichs
SB = bases.ShenBiharmonic
SN = bases.ShenNeumann
CN = bases.CombinedShenNeumann
MN = bases.MikNeumann
UD = bases.UpperDirichlet
LD = bases.LowerDirichlet
DN = bases.DirichletNeumann
ND = bases.NeumannDirichlet
CB = bases.CompositeBase
P1 = bases.Phi1
P2 = bases.Phi2
P3 = bases.Phi3
P4 = bases.Phi4

BCG = bases.BCGeneric

def get_ck(M, N, quad):
    """Return array ck, parameter in Chebyshev expansions

    Parameters
    ----------
    M : int
        The number of quadrature points in the test function
    N : int
        The number of quadrature points in the trial function
    quad : str
        Type of quadrature

        - GL - Chebyshev-Gauss-Lobatto
        - GC - Chebyshev-Gauss
    """
    ck = np.ones(min(M, N), int)
    ck[0] = 2
    if quad == "GL" and N >= M:
        ck[-1] = 2
    return ck

def dmax(N, M, d):
    Z = min(N, M)
    return Z-abs(d)+min(max((M-N)*int(d/abs(d)), 0), abs(d))

class BTTmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(T_j, T_k)_w,

    where :math:`T_k \in` :class:`.chebyshev.bases.Orthogonal`, and test
    and trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], T)
        assert isinstance(trial[0], T)
        ck = get_ck(test[0].N, trial[0].N, test[0].quad)
        SpectralMatrix.__init__(self, {0: np.pi/2*ck}, test, trial, scale=scale, measure=measure)
        self._matvec_methods += ['self']

    def matvec(self, v, c, format='csr', axis=0):
        c.fill(0)
        N, M = self.shape
        if not M == N:
            format = 'csr'
        if format == 'self':
            s = [np.newaxis,]*v.ndim # broadcasting
            d = tuple(slice(0, m) for m in v.shape)
            N, M = self.shape
            s[axis] = slice(0, M)
            s = tuple(s)
            c[d] = self[0][s]*v[d]
            self.scale_array(c, self.scale)
        else:
            format = None if format in self._matvec_methods else format
            c = super(BTTmat, self).matvec(v, c, format=format, axis=axis)

        return c


class BSDSDmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\phi_j, \phi_k)_w,

    where :math:`\phi_j \in` :class:`.chebyshev.bases.ShenDirichlet`, and test
    and trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        ck = get_ck(test[0].N, trial[0].N, test[0].quad)
        d = {0: np.pi/2*(ck[:-2]+ck[2:]),
             2: np.array([-np.pi/2])}
        d[-2] = d[2].copy()
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self._matvec_methods += ['cython', 'self']

    def get_solver(self):
        return generic_TDMA

    def matvec(self, v, c, format='cython', axis=0):
        N, M = self.shape
        c.fill(0)
        # Cython implementation only handles square matrix
        if not M == N:
            format = 'csr'

        if format == 'cython' and v.ndim == 3:
            ld = self[-2]*np.ones(M-2)
            cython.Matvec.Tridiagonal_matvec3D_ptr(v, c, ld, self[0], ld, axis)
            self.scale_array(c, self.scale)
        elif format == 'cython' and v.ndim == 2:
            ld = self[-2]*np.ones(M-2)
            cython.Matvec.Tridiagonal_matvec2D_ptr(v, c, ld, self[0], ld, axis)
            self.scale_array(c, self.scale)
        elif format == 'cython' and v.ndim == 1:
            ld = self[-2]*np.ones(M-2)
            cython.Matvec.Tridiagonal_matvec(v, c, ld, self[0], ld)
            self.scale_array(c, self.scale)
        elif format == 'self':
            if axis > 0:
                v = np.moveaxis(v, axis, 0)
                c = np.moveaxis(c, axis, 0)
            s = (slice(0, M),)+(np.newaxis,)*(v.ndim-1) # broadcasting
            sm2 = (slice(0, M-2),)+(np.newaxis,)*(v.ndim-1) # broadcasting
            c[:(M-2)] = self[2][sm2]*v[2:M]
            c[:M] += self[0][s]*v[:M]
            c[2:M] += self[-2][sm2]*v[:(M-2)]
            if axis > 0:
                v = np.moveaxis(v, 0, axis)
                c = np.moveaxis(c, 0, axis)
            self.scale_array(c, self.scale)

        else:
            format = None if format in ('cython', 'self') else format
            c = super(BSDSDmat, self).matvec(v, c, format=format, axis=axis)

        return c

class BSDSDmatW(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\phi_j, \phi_k (1-x^2))_w,

    where :math:`\phi_j \in` :class:`.chebyshev.bases.ShenDirichlet`, and test
    and trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        ck = get_ck(test[0].N, trial[0].N, test[0].quad)
        dk = np.ones(test[0].N)
        dk[:2] = 0
        d = {0: np.pi*((ck[:-2]+1)**2+1+dk[:-2])/8,
             2: -np.pi*(ck[:-4]+3)/8,
             4: np.pi/8}
        d[-2] = d[2].copy()
        d[-4] = d[4]
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

    def get_solver(self):
        return generic_PDMA

class BSNSDmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenNeumann`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SD)
        N = test[0].N
        M = trial[0].N
        Q = min(N, M)
        ck = get_ck(N, M, test[0].quad)
        k = np.arange(Q-2, dtype=float)
        d = {-2: -np.pi/2,
              0: np.pi/2.*(ck[:-2]+ck[2:]*(k/(k+2))**2)}
        d2 = -np.pi/2*(k/(k+2))**2
        d[2] = d2[:dmax(N-2, M-2, 2)].copy()
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class BSDSNmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenDirichlet`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.ShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SN)
        N = test[0].N
        M = trial[0].N
        Q = min(N, M)
        ck = get_ck(N, M, test[0].quad)
        k = np.arange(Q-2, dtype=float)
        d = {0:  np.pi/2.*(ck[:-2]+ck[2:]*(k/(k+2))**2),
             2: -np.pi/2}
        d[-2] = (-np.pi/2*(k/(k+2))**2)[:dmax(N-2, M-2, -2)]
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self._matvec_methods += ['cython']

    def matvec(self, v, c, format='cython', axis=0):
        c.fill(0)
        N, M = self.shape
        if not M == N:
            format = 'csr'
        if format == 'cython' and v.ndim == 3:
            cython.Matvec.BDN_matvec3D_ptr(v, c, self[-2], self[0], self[2], axis)
            self.scale_array(c, self.scale)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.BDN_matvec2D_ptr(v, c, self[-2], self[0], self[2], axis)
            self.scale_array(c, self.scale)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.BDN_matvec1D_ptr(v, c, self[-2], self[0], self[2])
            self.scale_array(c, self.scale)
        else:
            format = None if format in self._matvec_methods else format
            c = super(BSDSNmat, self).matvec(v, c, format=format, axis=axis)

        return c

class BLDLDmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\phi_j, \phi_k)_w,

    where :math:`\phi_k \in` :class:`.chebyshev.bases.LowerDirichlet`,
    and test and trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], LD)
        assert isinstance(trial[0], LD)
        ck = get_ck(test[0].N, trial[0].N, test[0].quad)
        d = {0: np.pi/2*(ck[:-1]+ck[1:]),
             1: np.array([np.pi/2]),
            -1: np.array([np.pi/2])}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

class BSNTmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(T_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenNeumann`, the
    trial :math:`T_j \in` :class:`.chebyshev.bases.Orthogonal`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], T)
        SpectralMatrix.__init__(self, {}, test, trial, scale=scale, measure=measure)


class BSNSBmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenNeumann`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.ShenBiharmonic`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SB)
        SpectralMatrix.__init__(self, {}, test, trial, scale=scale, measure=measure)

class BSNSNmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\phi_j, \phi_k)_w,

    where :math:`\phi_k \in` :class:`.chebyshev.bases.ShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SN)
        N = test[0].N
        M = trial[0].N
        Q = min(N, M)
        ck = get_ck(N, M, test[0].quad)
        k = np.arange(Q-2, dtype=float)
        d = {0: np.pi/2*(ck[:-2]+ck[2:]*(k[:]/(k[:]+2))**4)}
        dp = dmax(N-2, M-2, 2)
        d[2] = -np.pi/2*(k[:dp]/(k[:dp]+2))**2
        dm = dmax(N-2, M-2, -2)
        d[-2] = -np.pi/2*(k[:dm]/(k[:dm]+2))**2
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

    def get_solver(self):
        return generic_TDMA

    def matvec(self, v, c, format=None, axis=0):
        # Move axis to first
        if axis > 0:
            v = np.moveaxis(v, axis, 0)
            c = np.moveaxis(c, axis, 0)

        N = self.testfunction[0].N-2
        k = np.arange(N)
        d0 = self[0]
        d2 = self[2]
        c[0] = d0[0]*v[0] + d2[0]*v[2]
        c[1] = d0[1]*v[1] + d2[1]*v[3]
        for k in range(2, N-2):
            c[k] = d2[k-2]* v[k-2] + d0[k]*v[k] + d2[k]*v[k+2]

        c[N-2] = d2[N-4]*v[N-4] + d0[N-2]*v[N-2]
        c[N-1] = d2[N-3]*v[N-3] + d0[N-1]*v[N-1]
        c *= self.scale
        if axis > 0:
            v = np.moveaxis(v, 0, axis)
            c = np.moveaxis(c, 0, axis)
        return c


class BSDTmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(T_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenDirichlet`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.Orthogonal`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], T)
        N = test[0].N-2
        M = trial[0].N
        Q = min(N, M)
        ck = get_ck(test[0].N, trial[0].N, test[0].quad)
        d = {0: np.pi/2*ck[:Q],
             2: -np.pi/2*ck[2:(dmax(N, M, 2)+2)]}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class BTSDmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, T_k)_w,

    where the test function :math:`T_k \in` :class:`.chebyshev.bases.Orthogonal`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], T)
        assert isinstance(trial[0], SD)
        N = test[0].N
        M = trial[0].N-2
        Q = min(N, M)
        ck = get_ck(test[0].N, trial[0].N, test[0].quad)
        d = {0: np.pi/2*ck[:Q]}
        d[-2] = -np.pi/2
        if test[0].quad == 'GL':
            d[-2] = -np.pi/2*np.ones(Q)
            d[-2][-1] *= 2
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

class BTSNmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, T_k)_w,

    where the test function :math:`T_k \in` :class:`.chebyshev.bases.Orthogonal`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.ShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], T)
        assert isinstance(trial[0], SN)
        N = test[0].N
        ck = get_ck(N, trial[0].N, test[0].quad)
        k = np.arange(N, dtype=float)
        d = {-2: -np.pi/2*ck[2:]*((k[2:]-2)/k[2:])**2,
              0: np.pi/2*ck[:-2]}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

class BSBSBmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\phi_j, \phi_k)_w,

    where :math:`\phi_k \in` :class:`.chebyshev.bases.ShenBiharmonic`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        N = test[0].N
        M = trial[0].N
        Q = min(N, M)
        ck = get_ck(N, M, test[0].quad)
        k = np.arange(Q-4, dtype=float)
        d = {0: (ck[:Q-4] + 4*((k+2)/(k+3))**2 + ck[4:]*((k+1)/(k+3))**2)*np.pi/2.}
        d4 = (k+1)/(k+3)*np.pi/2
        d2 = -((k+2)/(k+3)+(k+4)*(k+1)/((k+5)*(k+3)))*np.pi
        d[2] = d2[:dmax(N-4, M-4, 2)]
        d[4] = d4[:dmax(N-4, M-4, 4)]
        d[-2] = d2[:dmax(N-4, M-4, -2)].copy()
        d[-4] = d4[:dmax(N-4, M-4, -4)].copy()
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self._matvec_methods += ['cython', 'self']

    def get_solver(self):
        return generic_PDMA

    def matvec(self, v, c, format='cython', axis=0):
        c.fill(0)
        N, M = self.shape
        if not M == N:
            format = 'csr'

        if format == 'self':
            if axis > 0:
                v = np.moveaxis(v, axis, 0)
                c = np.moveaxis(c, axis, 0)

            vv = v[:-4]
            s = (slice(None),) + (np.newaxis,)*(v.ndim-1) # broadcasting
            c[:N] = self[0][s] * vv[:]
            c[:N-2] += self[2][s] * vv[2:]
            c[:N-4] += self[4][s] * vv[4:]
            c[2:N] += self[-2][s] * vv[:-2]
            c[4:N] += self[-4][s] * vv[:-4]
            if axis > 0:
                v = np.moveaxis(v, 0, axis)
                c = np.moveaxis(c, 0, axis)
            self.scale_array(c, self.scale)

        elif format == 'cython' and v.ndim == 3:
            cython.Matvec.Pentadiagonal_matvec3D_ptr(v, c, self[-4], self[-2], self[0],
                                              self[2], self[4], axis)
            self.scale_array(c, self.scale)

        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.Pentadiagonal_matvec2D_ptr(v, c, self[-4], self[-2], self[0],
                                              self[2], self[4], axis)
            self.scale_array(c, self.scale)

        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.Pentadiagonal_matvec(v, c, self[-4], self[-2], self[0],
                                        self[2], self[4])
            self.scale_array(c, self.scale)
        else:
            format = None if format in self._matvec_methods else format
            c = super(BSBSBmat, self).matvec(v, c, format=format, axis=axis)

        return c


class BSBSDmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenBiharmonic`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SD)
        N = test[0].N-4
        M = trial[0].N-2
        Q = min(M, N)
        ck = get_ck(test[0].N, trial[0].N, test[0].quad)
        k = np.arange(Q, dtype=float)
        a = 2*(k+2)/(k+3)
        b = (k+1)/(k+3)
        d = {-2: -np.pi/2,
              0: (ck[:Q] + a)*np.pi/2,
              2: -(a[:min(Q, M-2)]+ck[4:min(Q+4, M+2)]*b[:min(Q, M-2)])*np.pi/2,
              4: b[:min(Q, M-4)]*ck[4:min(Q+4, M)]*np.pi/2}

        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self._matvec_methods += ['cython', 'self']

    def matvec(self, v, c, format='cython', axis=0):
        c.fill(0)
        N = self.shape[0]
        if format == 'self':
            if axis > 0:
                v = np.moveaxis(v, axis, 0)
                c = np.moveaxis(c, axis, 0)
            vv = v[:-2]
            s = (slice(None),) + (np.newaxis,)*(v.ndim-1) # broadcasting
            c[:N] = self[0][s] * vv[:-2]
            c[:N] += self[2][s] * vv[2:]
            c[:N-2] += self[4][s] * vv[4:]
            c[2:N] += self[-2] * vv[:-4]
            if axis > 0:
                c = np.moveaxis(c, 0, axis)
                v = np.moveaxis(v, 0, axis)
            self.scale_array(c, self.scale)

        elif format == 'cython' and v.ndim == 3:
            cython.Matvec.BBD_matvec3D_ptr(v, c, self[-2], self[0], self[2], self[4], axis)
            self.scale_array(c, self.scale)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.BBD_matvec2D_ptr(v, c, self[-2], self[0], self[2], self[4], axis)
            self.scale_array(c, self.scale)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.BBD_matvec1D_ptr(v, c, self[-2], self[0], self[2], self[4])
            self.scale_array(c, self.scale)
        else:
            format = None if format in self._matvec_methods else format
            c = super(BSBSDmat, self).matvec(v, c, format=format, axis=axis)

        return c

class BSBTmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(T_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenBiharmonic`, the
    trial :math:`T_j \in` :class:`.chebyshev.bases.Orthogonal`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], T)
        N, M = test[0].N-4, trial[0].N
        Q = min(M, N)
        ck = get_ck(test[0].N, trial[0].N, test[0].quad)
        k = np.arange(Q, dtype=float)
        d = {0: ck[:Q]*np.pi/2,
             2: -np.pi*(k[:min(Q, M-2)]+2)/(k[:min(Q, M-2)]+3),
             4: 0.5*np.pi*ck[4:min(Q+4, M)]*(k[:min(Q, M-4)]+1)/(k[:min(Q, M-4)]+3)}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

class BCNCNmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\phi_j, \phi_k)_w,

    where :math:`\phi_k \in` :class:`.chebyshev.bases.CombinedShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], CN)
        assert isinstance(trial[0], CN)
        assert trial[0].quad == 'GC', 'Not implemented for GL'
        N = test[0].N
        k = np.arange(N, dtype=float)
        dk = np.ones(N)
        dk[:3] = 0
        k2 = np.arange(N)
        k2[:3] = 3
        kk = np.arange(N)
        kk[0] = 1

        with np.errstate(invalid='ignore', divide='ignore'):
            d = {0: np.pi/2*((1+3*dk[:-2])/kk[:-2]**4 + 1/k[2:]**4 + dk[:-2]/(k2[:-2]-2)**4)}
            d[0][0] = np.pi
            d[2] = -np.pi/2*((1+dk[:-4])/k[:-4]**4 + 2/k[2:-2]**4)
            d[2][0] = 0
            d[4] = np.pi/2*dk[2:-4]/k[2:-4]**4
            d[-4] = d[4].copy()
            d[-2] = d[2].copy()

        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

    def get_solver(self):
        return generic_PDMA

class BSDHHmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenDirichlet`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.Heinrichs`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], HH)
        ck = get_ck(test[0].N, trial[0].N, test[0].quad)
        cp = np.ones(test[0].N); cp[2] = 2
        dk = np.ones(test[0].N); dk[:2] = 0
        d = {-2: -(np.pi/8)*ck[:-4],
             0: (np.pi/8)*(ck[:-2]*(ck[:-2]+ck[2:])+dk[:-2]),
             2: -(np.pi/8)*(ck[:-4]+2),
             4: (np.pi/8)}

        if trial[0].is_scaled():
            k = np.arange(test[0].N)
            d[-2] *= 1/(k[1:-3]*k[2:-2])
            d[0] *= 1/(k[1:-1]*k[2:])
            d[2] *= 1/((k[:-4]+3)*(k[:-4]+4))
            d[4] *= 1/((k[:-6]+5)*(k[:-6]+6))

        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

    def get_solver(self):
        return FDMA

class CSDSNmat(SpectralMatrix):
    r"""Derivative matrix :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(\psi'_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenDirichlet`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.ShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SN)
        N = test[0].N
        k = np.arange(N-2, dtype=float)
        d = {-1: -((k[1:]-1)/(k[1:]+1))**2*(k[1:]+1)*np.pi,
              1: (k[:-1]+1)*np.pi}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self._matvec_methods += ['cython']

    def matvec(self, v, c, format='cython', axis=0):
        c.fill(0)
        if format == 'cython' and v.ndim == 3:
            cython.Matvec.CDN_matvec3D_ptr(v, c, self[-1], self[1], axis)
            self.scale_array(c, self.scale)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.CDN_matvec2D_ptr(v, c, self[-1], self[1], axis)
            self.scale_array(c, self.scale)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.CDN_matvec1D_ptr(v, c, self[-1], self[1])
            self.scale_array(c, self.scale)
        else:
            format = None if format in self._matvec_methods else format
            c = super(CSDSNmat, self).matvec(v, c, format=format, axis=axis)

        return c


class CSDSDmat(SpectralMatrix):
    r"""Derivative matrix :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(\phi'_j, \phi_k)_w,

    where :math:`\phi_k \in` :class:`.chebyshev.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = {-1: -(k[1:N-2]+1)*np.pi,
              1: (k[:(N-3)]+1)*np.pi}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self._matvec_methods += ['cython', 'self']

    def matvec(self, v, c, format='cython', axis=0):
        N = self.shape[0]
        c.fill(0)
        if format == 'self':
            if axis > 0:
                v = np.moveaxis(v, axis, 0)
                c = np.moveaxis(c, axis, 0)

            s = (slice(None),) + (np.newaxis,)*(v.ndim-1) # broadcasting
            c[:N-1] = self[1][s]*v[1:N]
            c[1:N] += self[-1][s]*v[:(N-1)]
            if axis > 0:
                v = np.moveaxis(v, 0, axis)
                c = np.moveaxis(c, 0, axis)
            self.scale_array(c, self.scale)

        elif format == 'cython' and v.ndim == 3:
            cython.Matvec.CDD_matvec3D_ptr(v, c, self[-1], self[1], axis)
            self.scale_array(c, self.scale)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.CDD_matvec2D_ptr(v, c, self[-1], self[1], axis)
            self.scale_array(c, self.scale)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.CDD_matvec1D_ptr(v, c, self[-1], self[1])
            self.scale_array(c, self.scale)
        else:
            format = None if format in self._matvec_methods else format
            c = super(CSDSDmat, self).matvec(v, c, format=format, axis=axis)

        return c


class CSNSDmat(SpectralMatrix):
    r"""Derivative matrix :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(\psi'_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenNeumann`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N-2, dtype=float)
        self._keyscale = 1
        def _getkey(i):
            if i == -1:
                return -(k[1:]+1)*np.pi*self._keyscale
            elif i == 1:
                return -(2-k[:-1]**2/(k[:-1]+2)**2*(k[:-1]+3))*np.pi*self._keyscale
            return -(1-k[:-i]**2/(k[:-i]+2)**2)*2*np.pi*self._keyscale
        d = dict.fromkeys(np.arange(-1, N-1, 2), _getkey)
        #def _getkey(i):
        #    return -(1-k[:-i]**2/(k[:-i]+2)**2)*2*np.pi
        #d = dict.fromkeys(np.arange(-1, N-1, 2), _getkey)
        #d[-1] = -(k[1:]+1)*np.pi
        #d[1] = -(2-k[:-1]**2/(k[:-1]+2)**2*(k[:-1]+3))*np.pi
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class CTSDmat(SpectralMatrix):
    r"""Derivative matrix :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(\psi'_j, T_k)_w,

    where the test function :math:`T_k \in` :class:`.chebyshev.bases.Orthogonal`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], T)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = dict.fromkeys(np.arange(-1, N-2, 2), -2*np.pi)
        d[-1] = -(k[1:N-1]+1)*np.pi
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class ASBTmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (T''_j, \phi_k)_w

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenBiharmonic`, the trial
    function :math:`T_j \in` :class:`.chebyshev.bases.Orthogonal`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], T)
        N, M = test[0].N-4, trial[0].N
        Q = min(N, M)
        k = np.arange(Q, dtype=float)
        d = {2: 2*np.pi*(k[:min(Q, M-2)]+2)*(k[:min(Q, M-2)]+1)}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class CSDTmat(SpectralMatrix):
    r"""Derivative matrix :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(T'_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenDirichlet`, the
    trial :math:`T_j \in` :class:`.chebyshev.bases.Orthogonal`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], T)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = {1: np.pi*(k[:N-2]+1)}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

class CTTmat(SpectralMatrix):
    r"""Derivative matrix :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(T'_j, T_k)_w,

    where :math:`T_j \in` :class:`.chebyshev.bases.Orthogonal`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], T)
        assert isinstance(trial[0], T)
        N = test[0].N
        k = np.arange(N, dtype=float)
        self._keyscale = 1
        def _getkey(i):
            return np.pi*self._keyscale*k[i:]
        d = dict.fromkeys(np.arange(1, N, 2), _getkey)
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self._matvec_methods += ['cython']

    def matvec(self, v, c, format='cython', axis=0):
        c.fill(0)
        if format == 'cython' and v.ndim == 3:
            cython.Matvec.CTT_matvec3D_ptr(v, c, axis)
            self.scale_array(c, self.scale*self._keyscale)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.CTT_matvec2D_ptr(v, c, axis)
            self.scale_array(c, self.scale*self._keyscale)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.CTT_matvec(v, c)
            self.scale_array(c, self.scale*self._keyscale)
        else:
            format = None if format in self._matvec_methods else format
            c = super(CTTmat, self).matvec(v, c, format=format, axis=axis)
        return c


class CSBSDmat(SpectralMatrix):
    r"""Derivative matrix :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(\psi'_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenBiharmonic`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = {-1: -(k[1:N-4]+1)*np.pi,
              1: 2*(k[:N-4]+1)*np.pi,
              3: -(k[:N-5]+1)*np.pi}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self._matvec_methods += ['cython', 'self']

    def matvec(self, v, c, format='cython', axis=0):
        N, M = self.shape
        c.fill(0)
        if format == 'self':
            if axis > 0:
                c = np.moveaxis(c, axis, 0)
                v = np.moveaxis(v, axis, 0)
            s = (slice(None),) + (np.newaxis,)*(v.ndim-1) # broadcasting
            c[1:N] = self[-1][s]*v[:M-3]
            c[:N] += self[1][s]*v[1:M-1]
            c[:N-1] += self[3][s]*v[3:M]
            if axis > 0:
                c = np.moveaxis(c, 0, axis)
                v = np.moveaxis(v, 0, axis)
            self.scale_array(c, self.scale)
        elif format == 'cython' and v.ndim == 3:
            cython.Matvec.CBD_matvec3D_ptr(v, c, self[-1], self[1], self[3], axis)
            self.scale_array(c, self.scale)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.CBD_matvec2D_ptr(v, c, self[-1], self[1], self[3], axis)
            self.scale_array(c, self.scale)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.CBD_matvec(v, c, self[-1], self[1], self[3])
            self.scale_array(c, self.scale)
        else:
            format = None if format in self._matvec_methods else format
            c = super(CSBSDmat, self).matvec(v, c, format=format, axis=axis)
        return c


class CSDSBmat(SpectralMatrix):
    r"""Derivative matrix :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(\psi'_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenDirichlet`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.ShenBiharmonic`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SB)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = {-3: (k[3:-2]-2)*(k[3:-2]+1)/k[3:-2]*np.pi,
             -1: -2*(k[1:-3]+1)**2/(k[1:-3]+2)*np.pi,
              1: (k[:-5]+1)*np.pi}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self._matvec_methods += ['cython', 'self']

    def matvec(self, v, c, format='cython', axis=0):
        N, M = self.shape
        c.fill(0)
        if format == 'self':
            if axis > 0:
                c = np.moveaxis(c, axis, 0)
                v = np.moveaxis(v, axis, 0)

            s = (slice(None),) + (np.newaxis,)*(v.ndim-1) # broadcasting
            c[3:N] = self[-3][s] * v[:M-1]
            c[1:N-1] += self[-1][s] * v[:M]
            c[:N-3] += self[1][s] * v[1:M]
            if axis > 0:
                c = np.moveaxis(c, 0, axis)
                v = np.moveaxis(v, 0, axis)
            self.scale_array(c, self.scale)

        elif format == 'cython' and v.ndim == 3:
            cython.Matvec.CDB_matvec3D_ptr(v, c, self[-3], self[-1], self[1], axis)
            self.scale_array(c, self.scale)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.CDB_matvec2D_ptr(v, c, self[-3], self[-1], self[1], axis)
            self.scale_array(c, self.scale)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.CDB_matvec(v, c, self[-3], self[-1], self[1])
            self.scale_array(c, self.scale)
        else:
            format = None if format in self._matvec_methods else format
            c = super(CSDSBmat, self).matvec(v, c, format=format, axis=axis)

        return c


class ASBSBmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\phi''_j, \phi_k)_w

    where :math:`\phi_k \in` :class:`.chebyshev.bases.ShenBiharmonic`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        N = test[0].N
        k = np.arange(N-4, dtype=float)
        d = {-2: 2*(k[2:]-1)*(k[2:]+2)*np.pi,
              0: -4*((k+1)*(k+2)**2)/(k+3)*np.pi,
              2: 2*(k[:-2]+1)*(k[:-2]+2)*np.pi}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self._matvec_methods += ['cython', 'self']

    def matvec(self, v, c, format='cython', axis=0):
        N = self.shape[0]
        c.fill(0)
        if format == 'self':
            if axis > 0:
                c = np.moveaxis(c, axis, 0)
                v = np.moveaxis(v, axis, 0)

            s = (slice(None),) + (np.newaxis,)*(v.ndim-1) # broadcasting
            c[:N] = self[0][s] * v[:N]
            c[:N-2] += self[2][s] * v[2:N]
            c[2:N] += self[-2][s] * v[:N-2]
            if axis > 0:
                c = np.moveaxis(c, 0, axis)
                v = np.moveaxis(v, 0, axis)
            self.scale_array(c, self.scale)

        elif format == 'cython' and v.ndim == 3:
            cython.Matvec.Tridiagonal_matvec3D_ptr(v, c, self[-2], self[0], self[2], axis)
            self.scale_array(c, self.scale)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.Tridiagonal_matvec2D_ptr(v, c, self[-2], self[0], self[2], axis)
            self.scale_array(c, self.scale)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.Tridiagonal_matvec(v, c, self[-2], self[0], self[2])
            self.scale_array(c, self.scale)
        else:
            format = None if format in self._matvec_methods else format
            c = super(ASBSBmat, self).matvec(v, c, format=format, axis=axis)

        return c


class ASDSDmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\phi''_j, \phi_k)_w

    where :math:`\phi_k \in` :class:`.chebyshev.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=float)
        # use generator to save memory. Note that items are constant on a row for
        # keys > 2, which is taken advantage of in optimized matvecs and solvers.
        # These optimized versions never look up the diagonals for key > 2.
        self._keyscale = 1
        def _getkey(i):
            if i == 0:
                return -2*self._keyscale*np.pi*(k[:N-2]+1)*(k[:N-2]+2)
            elif i == 2:
                return -4*self._keyscale*np.pi*(k[:-4]+1)
            return -self._keyscale*4*np.pi*(k[:-(i+2)]+1)

        d = dict.fromkeys(np.arange(0, N-2, 2), _getkey)
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self._matvec_methods += ['cython']

    def get_solver(self):
        return ADDSolver

    def matvec(self, v, c, format='cython', axis=0):
        c.fill(0)
        if format == 'cython' and v.ndim == 3:
            cython.Matvec.ADD_matvec3D_ptr(v, c, self[0]/self._keyscale, axis)
            self.scale_array(c, self.scale*self._keyscale)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.ADD_matvec2D_ptr(v, c, self[0]/self._keyscale, axis)
            self.scale_array(c, self.scale*self._keyscale)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.ADD_matvec(v, c, self[0]/self._keyscale)
            self.scale_array(c, self.scale*self._keyscale)
        else:
            format = None if format in self._matvec_methods else format
            c = super(ASDSDmat, self).matvec(v, c, format=format, axis=axis)

        return c


class ASDSDmatW(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\phi''_j, (1-x^2)\phi_k)_w

    where :math:`\phi_k \in` :class:`.chebyshev.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = {0: -np.pi/2*(2*k[:-2]*k[2:]+6),
             2: np.pi/2*(k[:-4]+2)*(k[:-4]+3),
             -2: np.pi/2*k[2:-2]*(k[2:-2]-1)}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class ASNSNmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\phi''_j, \phi_k)_w

    where :math:`\phi_k \in` :class:`.chebyshev.bases.ShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SN)
        N = test[0].N
        k = np.arange(N-2, dtype=float)
        self._keyscale = 1
        def _getkey(i):
            if i == 0:
                return -2*np.pi*k**2*(k+1)/(k+2)*self._keyscale
            return -4*np.pi*(k[:-i]+i)**2*(k[:-i]+1)/(k[:-i]+2)**2*self._keyscale
        d = dict.fromkeys(np.arange(0, N-2, 2), _getkey)
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self._matvec_methods += ['numba']

    def get_solver(self):
        return ANNSolver

    def matvec(self, v, c, format=None, axis=0):
        if format == 'numba':
            try:
                c = numba.helmholtz.ANN_matvec(v, c, self, axis)
                return c
            except:
                pass

        # Move axis to first
        if axis > 0:
            v = np.moveaxis(v, axis, 0)
            c = np.moveaxis(c, axis, 0)
        N = self.testfunction[0].N-2
        k = np.arange(N)
        j2 = k**2
        if v.ndim > 1:
            s = [np.newaxis]*v.ndim
            s[0] = slice(None)
            j2 = j2[tuple(s)]
        vc = v[:-2]*j2
        d0 = -2*np.pi*(k+1)/(k+2)*self.scale*self._keyscale
        d2 = -4*np.pi*(k[:-2]+1)/(k[:-2]+2)**2*self.scale*self._keyscale
        c[N-1] = d0[N-1]*vc[N-1]
        c[N-2] = d0[N-2]*vc[N-2]
        s0 = 0
        s1 = 0
        for k in range(N-3, -1, -1):
            c[k] = d0[k]*vc[k]
            if k % 2 == 0:
                s0 += vc[k+2]
                c[k] += s0*d2[k]
            else:
                s1 += vc[k+2]
                c[k] += s1*d2[k]
        c *= self.scale
        if axis > 0:
            v = np.moveaxis(v, 0, axis)
            c = np.moveaxis(c, 0, axis)
        return c

class ASBSDmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\psi''_j, \phi_k)_w

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenBiharmonic`,
    the trial function :math:`\psi_j \in` :class:`.chebyshev.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SD)
        N = test[0].N-4
        M = trial[0].N-2
        Q = min(M, N)
        k = np.arange(Q, dtype=float)
        d = {0: -2*(k+1)*(k+2)*np.pi,
             2: 2*(k[:min(Q, M-2)]+1)*(k[:min(Q, M-2)]+2)*np.pi}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class ATTmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (T''_j, T_k)_w

    where :math:`T_k \in` :class:`.chebyshev.bases.Orthogonal`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], T)
        assert isinstance(trial[0], T)
        N = test[0].N
        k = np.arange(N, dtype=float)
        self._keyscale = 1
        def _getkey(j):
            return self._keyscale*k[j:]*(k[j:]**2-k[:-j]**2)*np.pi/2
        d = dict.fromkeys(np.arange(2, N, 2), _getkey)
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self._matvec_methods += ['cython']

    def matvec(self, v, c, format='cython', axis=0):
        c.fill(0)
        if format == 'cython' and v.ndim == 3:
            cython.Matvec.ATT_matvec3D_ptr(v, c, axis)
            self.scale_array(c, self.scale*self._keyscale)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.ATT_matvec2D_ptr(v, c, axis)
            self.scale_array(c, self.scale*self._keyscale)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.ATT_matvec(v, c)
            self.scale_array(c, self.scale*self._keyscale)
        else:
            format = None if format in self._matvec_methods else format
            c = super(ATTmat, self).matvec(v, c, format=format, axis=axis)
        return c

class ATSDmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\psi''_j, T_k)_w

    where the test function :math:`T_k \in` :class:`.chebyshev.bases.Orthogonal`,
    the trial function :math:`\psi_j \in` :class:`.chebyshev.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], T)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=float)
        self._keyscale = 1
        def _getkey(j):
            if j == 0:
                return -np.pi/2*k[2:]*(k[2:]**2-k[:-2]**2)*self._keyscale
            return (k[j:-2]*(k[j:-2]**2-k[:-(j+2)]**2) -
                    k[j+2:]*(k[j+2:]**2-k[:-(j+2)]**2))*np.pi/2.*self._keyscale
        d = dict.fromkeys(np.arange(0, N-2, 2), _getkey)
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

class ATSNmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\psi''_j, T_k)_w

    where the test function :math:`T_k \in` :class:`.chebyshev.bases.Orthogonal`,
    the trial function :math:`\psi_j \in` :class:`.chebyshev.bases.ShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], T)
        assert isinstance(trial[0], SN)
        N = test[0].N
        k = np.arange(N, dtype=float)
        self._keyscale = 1
        def _getkey(j):
            if j == 0:
                return -k[:-2]**2/(k[:-2]+2)*((k[:-2]+2)**2-k[:-2]**2)*np.pi/2.*self._keyscale
            return (k[j:-2]*(k[j:-2]**2-k[:-(j+2)]**2) - k[j:-2]**2/(k[j:-2]+2)*((k[j:-2]+2)**2-k[:-(j+2)]**2))*np.pi/2.*self._keyscale
        d = dict.fromkeys(np.arange(0, N-2, 2), _getkey)
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

class ACNCNmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\phi''_j, \phi_k)_w

    where :math:`\phi_k \in` :class:`.chebyshev.bases.CombinedShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], CN)
        assert isinstance(trial[0], CN)
        N = test[0].N
        k = np.arange(N-2, dtype=float)
        with np.errstate(invalid='ignore', divide='ignore'):
            d = {-2: 2*np.pi*(k[2:]-1)/k[2:]/(k[2:]-2)**2}
            d[0] = -2*np.pi*((k-1)/k**2/(k-2)+(k+1)/k**2/(k+2))
            d[2] = 2*np.pi*(k[:-2]+1)/k[:-2]/k[2:]**2
            d[0][:3] = -2*np.pi/k[:3]**2*((k[:3]+1)/(k[:3]+2))
        d[0][0] = 0
        d[-2][0] = 0
        d[2][0] = -np.pi
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

    def get_solver(self):
        return generic_TDMA


class ASDHHmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\psi''_j, \phi_k)_w

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenDirichlet`,
    the trial function :math:`\psi_j \in` :class:`.chebyshev.bases.Heinrichs`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], HH)
        N = test[0].N
        k = np.arange(N)
        ck = get_ck(N, trial[0].N, test[0].quad)

        if trial[0].is_scaled():
            d = {0: -np.pi/2*ck[:-2],
                 2: np.pi/2*(k[:-4]*k[1:-3])/((k[:-4]+3)*(k[:-4]+4))}
        else:
            d = {0: -np.pi/2*ck[:-2]*k[2:]*k[1:-1],
                 2: np.pi/2*k[:-4]*k[1:-3]}

        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

    def get_solver(self):
        return TwoDMA

class AHHHHmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\phi''_j, \phi_k)_w

    where :math:`\phi_k \in` :class:`.chebyshev.bases.Heinrichs`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], HH)
        assert isinstance(trial[0], HH)
        assert test[0].is_scaled() == trial[0].is_scaled()
        N = test[0].N
        k = np.arange(N-2, dtype=float)
        ck = get_ck(N, trial[0].N, test[0].quad)
        dk = ck.copy()
        dk[:2] = 0
        d = {0: -np.pi/8*ck[:-2]**2*((k+1)*(k+2)+dk[:-2]*(k-2)*(k-1)),
             2: np.pi/8*ck[:-4]*k[:-2]*k[1:-1],
            -2: np.pi/8*ck[:-4]*k[2:]*k[1:-1]}
        if test[0].is_scaled() and trial[0].is_scaled():
            d[0] *= 1/((k+2)**2*(k+1)**2)
            d[2] *= 1/((k[:-2]+1)*(k[:-2]+2)*(k[:-2]+3)*(k[:-2]+4))
            d[-2]*= 1/(k[1:-1]*k[2:]*(k[2:]+1)*(k[2:]+2))

        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

class BHHHHmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\phi_j, \phi_k)_w,

    where :math:`\phi_k \in` :class:`.chebyshev.bases.Heinrichs`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], HH)
        assert isinstance(trial[0], HH)
        ck = get_ck(test[0].N, trial[0].N, test[0].quad)
        cp = np.ones(test[0].N-2); cp[2] = 2
        dk = np.ones(test[0].N-2); dk[:2] = 0
        d = {0: np.pi/16*(ck[:-2]**2*(ck[:-2]+ck[2:])/2 + dk*(cp+3)/2),
             2: -np.pi/16*(ck[:-4]+ck[:-4]**2/2+dk[:-2]/2),
             4: np.pi*ck[:-6]/32}

        if test[0].is_scaled() and trial[0].is_scaled():
            k = np.arange(test[0].N-2)
            d[0] *= 1/((k+2)**2*(k+1)**2)
            d[2] *= 1/((k[:-2]+1)*(k[:-2]+2)*(k[:-2]+3)*(k[:-2]+4))
            d[4] *= 1/((k[:-4]+1)*(k[:-4]+2)*(k[:-4]+5)*(k[:-4]+6))
        d[-2] = d[2].copy()
        d[-4] = d[4].copy()
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

    def get_solver(self):
        return generic_PDMA

class BSDMNmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenDirichlet`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.MikNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], MN)
        N = test[0].N
        k = np.arange(N, dtype=float)
        ck = get_ck(test[0].N, trial[0].N, test[0].quad)
        kp = k.copy(); kp[0] = 1
        qk = np.ones(N)
        qk[0] = 0
        qk[1] = 1.5
        dk = np.ones(N)
        dk[0] = 0
        d = {-2: -np.pi/2/k[2:-2]/k[1:-3],
              0: (np.pi/2)*((2*qk[:-2]/kp[:-2] + ck[2:]/k[2:]))/k[1:-1],
              2: -(np.pi/2)*((1/kp[:-4] + 2/k[2:-2]))/k[3:-1],
              4: (np.pi/2)/k[2:-4]/k[5:-1]}
        d[-2][0] = 0
        d[0][0] = np.pi
        d[2][0] = -np.pi/6
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

class ASDMNmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\psi''_j, \phi_k)_w

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenDirichlet`,
    the trial function :math:`\psi_j \in` :class:`.chebyshev.bases.MikNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], MN)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = {0: -2*np.pi*np.ones(N-2),
             2: 2*np.pi*k[1:-3]/k[3:-1]}
        d[0][0] = 0
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

    def get_solver(self):
        return TwoDMA

class ASNCNmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\psi''_j, \phi_k)_w

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenNeumann`,
    the trial function :math:`\psi_j \in` :class:`.chebyshev.bases.CombinedShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], CN)
        N = test[0].N
        k = np.arange(N, dtype=float)
        qk = np.ones(N)
        qk[0] = 0
        qk[1] = 1.5
        dk = np.ones(N)
        dk[0] = 0
        d = {0: -2*np.pi*k[1:-1]/k[2:],
             2: 2*np.pi*k[:-4]*k[1:-3]/k[2:-2]**2}
        d[0][0] = 0
        d[2][0] = -np.pi

        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

    def get_solver(self):
        return TwoDMA



class SSBSBmat(SpectralMatrix):
    r"""Biharmonic matrix :math:`S=(s_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        s_{kj}=(\phi''''_j, \phi_k)_w,

    where :math:`\phi_k \in` :class:`.chebyshev.bases.ShenBiharmonic`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        N = test[0].N
        ki = np.arange(N-4)
        k = np.arange(N-4, dtype=float)
        self._keyscale = 1
        def _getkey(j):
            if j == 0:
                return self._keyscale*np.pi*8*(ki+1)**2*(ki+2)*(ki+4)
            else:
                i = 8*self._keyscale*(ki[:-j]+1)*(ki[:-j]+2)*(ki[:-j]*(ki[:-j]+4)+3*(ki[j:]+2)**2)
                return np.array(i*np.pi/(k[j:]+3))
        d = dict.fromkeys(np.arange(0, N-4, 2), _getkey)
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self._matvec_methods += ['cython']

    def matvec(self, v, c, format='cython', axis=0):
        c.fill(0)
        if format == 'cython' and v.ndim == 3:
            cython.Matvec.SBB_matvec3D_ptr(v, c, self[0]/self._keyscale, axis)
            self.scale_array(c, self.scale*self._keyscale)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.SBB_matvec2D_ptr(v, c, self[0]/self._keyscale, axis)
            self.scale_array(c, self.scale*self._keyscale)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.SBBmat_matvec(v, c, self[0]/self._keyscale)
            self.scale_array(c, self.scale*self._keyscale)
        else:
            format = None if format in self._matvec_methods else format
            c = super(SSBSBmat, self).matvec(v, c, format=format, axis=axis)

        return c

class BGBCGmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, \phi_k)_w,

    where the test function :math:`\phi_k` is a subclass of
    :class:`.chebyshev.bases.CompositeBase`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.BCGeneric`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], CB)
        assert isinstance(trial[0], BCG)
        B = BTTmat((test[0].get_orthogonal(domain=(-1, 1)), 0), (trial[0].get_orthogonal(domain=(-1, 1)), 0))
        K = test[0].stencil_matrix()
        K.shape = (test[0].dim(), test[0].N)
        S = extract_diagonal_matrix(trial[0].stencil_matrix().T).diags('csr')
        q = sp.degree(measure)
        if measure != 1:
            from shenfun.jacobi.recursions import pmat, half, cn, a
            assert sp.sympify(measure).is_polynomial()

            A = sp.S(0)
            for dv in split(measure, expand=True):
                sc = dv['coeff']
                msi = dv['x']
                qi = sp.degree(msi)
                Ax = pmat(a, qi, -half, -half, test[0].N, test[0].N, cn)
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
    any subclasses of :class:`.chebyshev.bases.CompositeBase` and :math:`q \ge 0`
    is an integer. Test and trial spaces have dimensions of M and N, respectively.

    Note
    ----
    Creating mass matrices this way is efficient in terms of memory since the
    mass matrix of the orthogonal basis is diagonal.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], CB)
        assert isinstance(trial[0], CB)
        B = BTTmat((test[0].get_orthogonal(domain=(-1, 1)), 0), (trial[0].get_orthogonal(domain=(-1, 1)), 0))
        K = test[0].stencil_matrix()
        K.shape = (test[0].dim(), test[0].N)
        S = trial[0].stencil_matrix()
        S.shape = (trial[0].dim(), trial[0].N)
        q = sp.degree(measure)
        if measure != 1:
            from shenfun.jacobi.recursions import pmat, half, cn, a
            assert sp.sympify(measure).is_polynomial()
            A = sp.S(0)
            for dv in split(measure, expand=True):
                sc = dv['coeff']
                msi = dv['x']
                qi = sp.degree(msi)
                Ax = pmat(a, qi, -half, -half, test[0].N, test[0].N, cn)
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

    where the test function :math:`\phi_i` is in one of :class:`.chebyshev.bases.Phi1`,
    :class:`.chebyshev.bases.Phi2`, :class:`.chebyshev.bases.Phi3`, :class:`.chebyshev.bases.Phi4`,
    the trial :math:`\psi_j` any class in :class:`.chebyshev.bases`,
    The three parameters k, q and l are integers, and test and trial spaces
    have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], (P1, P2, P3, P4))
        assert test[0].quad == 'GC'
        from shenfun.jacobi.recursions import Lmat, half, cn
        q = sp.degree(measure)
        k = (test[0].N-test[0].dim())//2
        l = k-trial[1]
        if q > 0 and test[0].domain != test[0].reference_domain():
            D = sp.S(0)
            for dv in split(measure, expand=True):
                sc = dv['coeff']
                msi = dv['x']
                qi = sp.degree(msi)
                Ax = Lmat(k, qi, l, test[0].dim(), trial[0].N, -half, -half, cn)
                D = D + sc*Ax
        else:
            D = Lmat(k, q, l, test[0].dim(), trial[0].N, -half, -half, cn)

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

    where the test function :math:`\phi_i` is in one of :class:`.chebyshev.bases.Phi1`,
    :class:`.chebyshev.bases.Phi2`, :class:`.chebyshev.bases.Phi3`, :class:`.chebyshev.bases.Phi4`,
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.BCGeneric`.
    The three parameters k, q, l are integers and test and trial spaces have
    dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], (P1, P2, P3, P4))
        assert isinstance(trial[0], BCG)
        from shenfun.jacobi.recursions import Lmat, half, cn
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
                Ax = Lmat(k, qi, l, M, N, -half, -half, cn)
                D = D + sc*Ax
        else:
            D = Lmat(k, q, l, M, N, -half, -half, cn)

        K = trial[0].stencil_matrix()
        D = extract_diagonal_matrix(D*extract_diagonal_matrix(K).diags('csr').T, lowerband=N+q, upperband=N)
        SpectralMatrix.__init__(self, dict(D), test, trial, scale=scale, measure=measure)


class _Chebmatrix(SpectralMatrix):
    def __init__(self, test, trial, scale=1, measure=1):
        SpectralMatrix.__init__(self, {}, test, trial, scale=scale, measure=measure)


class _ChebMatDict(dict):
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
            c = functools.partial(_Chebmatrix, measure=measure)
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
mat = _ChebMatDict({
    ((T,  0), (T , 0)): BTTmat,
    ((SD, 0), (SD, 0)): BSDSDmat,
    ((SD, 0), (SD, 0), (1-x**2)): BSDSDmatW,
    ((SB, 0), (SB, 0)): BSBSBmat,
    ((SN, 0), (SN, 0)): BSNSNmat,
    ((CN, 0), (CN, 0)): BCNCNmat,
    ((SN, 0), (T , 0)): BSNTmat,
    ((SN, 0), (SB, 0)): BSNSBmat,
    ((SD, 0), (SN, 0)): BSDSNmat,
    ((SN, 0), (SD, 0)): BSNSDmat,
    ((LD, 0), (LD, 0)): BLDLDmat,
    ((T,  0), (SN, 0)): BTSNmat,
    ((SB, 0), (SD, 0)): BSBSDmat,
    ((SB, 0), (T,  0)): BSBTmat,
    ((T,  0), (SD, 0)): BTSDmat,
    ((SD, 0), (T,  0)): BSDTmat,
    ((SD, 0), (SD, 2)): ASDSDmat,
    ((SD, 0), (SD, 2), (1-x**2)): ASDSDmatW,
    ((T,  0), (T , 2)): ATTmat,
    ((T,  0), (SD, 2)): ATSDmat,
    ((T,  0), (SN, 2)): ATSNmat,
    ((SN, 0), (SN, 2)): ASNSNmat,
    ((CN, 0), (CN, 2)): ACNCNmat,
    ((SB, 0), (SB, 2)): ASBSBmat,
    ((SB, 0), (SD, 2)): ASBSDmat,
    ((HH, 0), (HH, 2)): AHHHHmat,
    ((HH, 0), (HH, 0)): BHHHHmat,
    ((SD, 0), (MN, 0)): BSDMNmat,
    ((SD, 0), (MN, 2)): ASDMNmat,
    ((SN, 0), (CN, 2)): ASNCNmat,
    ((SN, 0), (CN, 0)): BGGmat,
    ((SD, 0), (HH, 0)): BSDHHmat,
    ((SD, 0), (HH, 2)): ASDHHmat,
    ((SB, 0), (SB, 4)): SSBSBmat,
    ((SD, 0), (SN, 1)): CSDSNmat,
    ((SB, 0), (SD, 1)): CSBSDmat,
    ((SB, 0), (T,  2)): ASBTmat,
    ((T,  0), (SD, 1)): CTSDmat,
    ((T,  0), (T,  1)): CTTmat,
    ((SD, 0), (SD, 1)): CSDSDmat,
    ((SN, 0), (SD, 1)): CSNSDmat,
    ((SD, 0), (SB, 1)): CSDSBmat,
    ((SD, 0), (T,  1)): CSDTmat,
    ((SD, 0), (BCG, 0)): BGBCGmat,
    ((HH, 0), (BCG, 0)): BGBCGmat,
    ((SN, 0), (BCG, 0)): BGBCGmat,
    ((SB, 0), (BCG, 0)): BGBCGmat,
    ((CN, 0), (BCG, 0)): BGBCGmat,
    ((MN, 0), (BCG, 0)): BGBCGmat,
    ((UD, 0), (BCG, 0)): BGBCGmat,
    ((LD, 0), (BCG, 0)): BGBCGmat,
    ((DN, 0), (BCG, 0)): BGBCGmat,
    ((ND, 0), (BCG, 0)): BGBCGmat,
    ('PX', 0): PXGmat,
    ('PX', 1): PXBCGmat,
    })

#mat = _ChebMatDict({})
