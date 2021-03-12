r"""
This module contains specific inner product matrices for the different bases in
the Chebyshev family.

A naming convention is used for the first capital letter for all matrices.
The first letter refers to type of matrix.

    - Mass matrices start with `B`
    - One derivative start with `C`
    - Two derivatives (Laplace) start with `A`
    - Four derivatives (Biharmonic) start with `S`

A matrix may consist of different types of test and trialfunctions as long as
they are all in the Chebyshev family. The next letters in the matrix name uses
the short form for all these different bases according to

T  = Orthogonal
SD = ShenDirichlet
HH = Heinrichs
SB = ShenBiharmonic
SN = ShenNeumann
CN = CombinedShenNeumann
MN = MikNeumann
TU = MixedTU
DU = DirichletU
UD = UpperDirichlet
DN = DirichletNeumann
BD = BCDirichlet
BB = BCBiharmonic

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
from shenfun.matrixbase import SpectralMatrix
from shenfun.la import TDMA as generic_TDMA
from shenfun.la import PDMA as generic_PDMA
from .la import TDMA, PDMA, TwoDMA, FDMA
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
TU = bases.MixedTU
DU = bases.DirichletU
UD = bases.UpperDirichlet
DN = bases.DirichletNeumann

BCD = bases.BCDirichlet
BCB = bases.BCBiharmonic

def get_ck(N, quad):
    """Return array ck, parameter in Chebyshev expansions

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str
               Type of quadrature

               - GL - Chebyshev-Gauss-Lobatto
               - GC - Chebyshev-Gauss
    """
    ck = np.ones(N, int)
    ck[0] = 2
    if quad == "GL":
        ck[-1] = 2
    return ck

def dmax(N, M, d):
    Z = min(N, M)
    return Z-abs(d)+min(max((M-N)*int(d/abs(d)), 0), abs(d))


class BSDSDmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        B_{kj}=(\phi_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\phi_j` is a Shen Dirichlet basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        ck = get_ck(test[0].N, test[0].quad)
        d = {0: np.pi/2*(ck[:-2]+ck[2:]),
             2: np.array([-np.pi/2])}
        d[-2] = d[2]
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
        self.solve = TDMA(self)
        self._matvec_methods += ['cython', 'self']

    def matvec(self, v, c, format='cython', axis=0):
        N, M = self.shape
        c.fill(0)
        # Cython implementation only handles square matrix
        if not M == N:
            format = 'csr'

        if format == 'cython' and v.ndim == 3:
            ld = self[-2]*np.ones(M-2)
            cython.Matvec.Tridiagonal_matvec3D_ptr(v, c, ld, self[0], ld, axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 2:
            ld = self[-2]*np.ones(M-2)
            cython.Matvec.Tridiagonal_matvec2D_ptr(v, c, ld, self[0], ld, axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 1:
            ld = self[-2]*np.ones(M-2)
            cython.Matvec.Tridiagonal_matvec(v, c, ld, self[0], ld)
            self.scale_array(c)
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
            self.scale_array(c)

        else:
            c = super(BSDSDmat, self).matvec(v, c, format=format, axis=axis)

        return c

class BSDSDmatW(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        B_{kj}=(\phi_j, \phi_k (x^2-1))_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\phi_j` is a Shen Dirichlet basis function.
    weight = 1-x**2

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        ck = get_ck(test[0].N, test[0].quad)
        dk = np.ones(test[0].N)
        dk[:2] = 0
        d = {0: np.pi*((ck[:-2]+1)**2+1+dk[:-2])/8,
             2: -np.pi*(ck[:-4]+3)/8,
             4: np.pi/8}
        d[-2] = d[2]
        d[-4] = d[4]
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        #self.solve = PDMA(self)


class BSNSDmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\phi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 1, 2, ..., N-2

    :math:`\psi_k` is the Shen Dirichlet basis function and :math:`\phi_j` is a
    Shen Neumann basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SD)
        N = test[0].N
        M = trial[0].N
        Q = min(N, M)
        ck = get_ck(Q, test[0].quad)
        k = np.arange(Q-2, dtype=float)
        d = {-2: -np.pi/2,
              0: np.pi/2.*(ck[:-2]+ck[2:]*(k/(k+2))**2)}
        d2 = -np.pi/2*(k/(k+2))**2
        d[2] = d2[:dmax(N-2, M-2, 2)]
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)

    def matvec(self, v, c, format='csr', axis=0):
        c = super(BSNSDmat, self).matvec(v, c, format=format, axis=axis)
        s = [slice(None),]*v.ndim
        if self.testfunction[0].use_fixed_gauge:
            s[axis] = 0
            c[tuple(s)] = 0
        return c


class BSDSNmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\psi_j, \phi_k)_w

    where

    .. math::

        j = 1, 2, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    :math:`\psi_j` is the Shen Dirichlet basis function and :math:`\phi_k` is a
    Shen Neumann basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SN)
        N = test[0].N
        M = trial[0].N
        Q = min(N, M)
        ck = get_ck(Q, test[0].quad)
        k = np.arange(Q-2, dtype=float)
        d = {0:  np.pi/2.*(ck[:-2]+ck[2:]*(k/(k+2))**2),
             2: -np.pi/2}
        d[-2] = (-np.pi/2*(k/(k+2))**2)[:dmax(N-2, M-2, -2)]
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
        self._matvec_methods += ['cython']

    def matvec(self, v, c, format='cython', axis=0):
        c.fill(0)
        N, M = self.shape
        if not M == N:
            format = 'csr'
        if format == 'cython' and v.ndim == 3:
            cython.Matvec.BDN_matvec3D_ptr(v, c, self[-2], self[0], self[2], axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.BDN_matvec2D_ptr(v, c, self[-2], self[0], self[2], axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.BDN_matvec1D_ptr(v, c, self[-2], self[0], self[2])
            self.scale_array(c)
        else:
            c = super(BSDSNmat, self).matvec(v, c, format=format, axis=axis)

        return c


class BSNTmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (T_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 1, 2, ..., N-2

    :math:`\phi_k` is the Shen Neumann basis function and :math:`T_j` is a
    Chebyshev basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], T)
        SpectralMatrix.__init__(self, {}, test, trial)

    def matvec(self, v, c, format='csr', axis=0):
        c = super(BSNTmat, self).matvec(v, c, format=format, axis=axis)
        s = [slice(None),]*v.ndim
        if self.testfunction[0].use_fixed_gauge:
            s[axis] = 0
            c[tuple(s)] = 0
        return c

class BSNSBmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\phi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-4 \text{ and } k = 1, 2, ..., N-2

    :math:`\psi_k` is the Shen Neumann basis function and :math:`\phi_j` is a
    Shen biharmonic basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SB)
        SpectralMatrix.__init__(self, {}, test, trial)

    def matvec(self, v, c, format='csr', axis=0):
        c = super(BSNSBmat, self).matvec(v, c, format=format, axis=axis)
        s = [slice(None),]*v.ndim
        if self.testfunction[0].use_fixed_gauge:
            s[axis] = 0
            c[tuple(s)] = 0
        return c


class BTTmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (T_j, T_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N

    and :math:`T_j` is the jth order Chebyshev function of the first kind.
    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], T)
        assert isinstance(trial[0], T)
        ck = get_ck(min(test[0].N, trial[0].N), test[0].quad)
        SpectralMatrix.__init__(self, {0: np.pi/2*ck}, test, trial)
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
            self.scale_array(c)
        else:
            c = super(BTTmat, self).matvec(v, c, format=format, axis=axis)

        return c

    def solve(self, b, u=None, axis=0):
        s = self.trialfunction[0].slice()
        if u is None:
            u = b
        else:
            assert u.shape == b.shape
        sl = [np.newaxis]*u.ndim
        sl[axis] = s
        sl = tuple(sl)
        ss = self.trialfunction[0].sl[s]
        d = (1./self.scale)/self[0]
        u[ss] = b[ss]*d[sl]
        return u

class BSNSNmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\phi_j, \phi_k)_w

    where

    .. math::

        j = 1, 2, ..., N-2 \text{ and } k = 1, 2, ..., N-2

    and :math:`\phi_j` is the Shen Neumann basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SN)
        N = test[0].N
        M = trial[0].N
        Q = min(N, M)
        ck = get_ck(Q, test[0].quad)
        k = np.arange(Q-2, dtype=float)
        d = {0: np.pi/2*(ck[:-2]+ck[2:]*(k[:]/(k[:]+2))**4)}

        dp = dmax(N-2, M-2, 2)
        d[2] = -np.pi/2*(k[:dp]/(k[:dp]+2))**2
        dm = dmax(N-2, M-2, -2)
        d[-2] = -np.pi/2*(k[:dm]/(k[:dm]+2))**2
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
        self.solve = generic_TDMA(self)

    def matvec(self, v, c, format='csr', axis=0):
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
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (T_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N-2

    :math:`\phi_k` is the Shen Dirichlet basis function and :math:`T_j` is the
    Chebyshev basis function.
    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], T)
        N = test[0].N-2
        M = trial[0].N
        Q = min(N, M)
        ck = get_ck(Q+2, test[0].quad)
        d = {0: np.pi/2*ck[:Q],
             2: -np.pi/2*ck[2:(dmax(N, M, 2)+2)]}
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)


class BTSDmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\phi_j, T_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N

    :math:`\phi_j` is the Shen Dirichlet basis function and :math:`T_k` is the
    Chebyshev basis function.
    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], T)
        assert isinstance(trial[0], SD)
        N = test[0].N
        M = trial[0].N-2
        Q = min(N, M)
        ck = get_ck(N, test[0].quad)
        d = {0: np.pi/2*ck[:Q]}
        d[-2] = -np.pi/2*ck[2:(dmax(N, M, -2)+2)]
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)


class BTSNmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\phi_j, T_k)_w

    where

    .. math::

        j = 1, 2, ..., N-2 \text{ and } k = 0, 1, ..., N

    :math:`\phi_j` is the Shen Neumann basis function and :math:`T_k` is the
    Chebyshev basis function.
    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], T)
        assert isinstance(trial[0], SN)
        N = test[0].N
        ck = get_ck(N, test[0].quad)
        k = np.arange(N, dtype=float)
        d = {-2: -np.pi/2*ck[2:]*((k[2:]-2)/k[2:])**2,
              0: np.pi/2*ck[:-2]}
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)


class BSBSBmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\psi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-4 \text{ and } k = 0, 1, ..., N-4

    and :math:`\psi_j` is the Shen Biharmonic basis function.
    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        from shenfun.la import PDMA
        N = test[0].N
        M = trial[0].N
        Q = min(N, M)
        ck = get_ck(Q, test[0].quad)
        k = np.arange(Q-4, dtype=float)
        d = {0: (ck[:Q-4] + 4*((k+2)/(k+3))**2 + ck[4:]*((k+1)/(k+3))**2)*np.pi/2.}
        d4 = (k+1)/(k+3)*np.pi/2
        d2 = -((k+2)/(k+3)+(k+4)*(k+1)/((k+5)*(k+3)))*np.pi
        d[2] = d2[:dmax(N-4, M-4, 2)]
        d[4] = d4[:dmax(N-4, M-4, 4)]
        d[-2] = d2[:dmax(N-4, M-4, -2)]
        d[-4] = d4[:dmax(N-4, M-4, -4)]
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
        self.solve = PDMA(self)
        self._matvec_methods += ['cython', 'self']

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
            self.scale_array(c)

        elif format == 'cython' and v.ndim == 3:
            cython.Matvec.Pentadiagonal_matvec3D_ptr(v, c, self[-4], self[-2], self[0],
                                              self[2], self[4], axis)
            self.scale_array(c)

        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.Pentadiagonal_matvec2D_ptr(v, c, self[-4], self[-2], self[0],
                                              self[2], self[4], axis)
            self.scale_array(c)

        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.Pentadiagonal_matvec(v, c, self[-4], self[-2], self[0],
                                        self[2], self[4])
            self.scale_array(c)
        else:
            c = super(BSBSBmat, self).matvec(v, c, format=format, axis=axis)

        return c


class BSBSDmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\phi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-4

    and :math:`\phi_j` is the Shen Dirichlet basis function and :math:`\psi_k`
    the Shen Biharmonic basis function.
    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SD)
        N = test[0].N
        ck = get_ck(N, test[0].quad)
        k = np.arange(N-4, dtype=float)
        a = 2*(k+2)/(k+3)
        b = (k[:N-4]+1)/(k[:N-4]+3)
        d = {-2: -np.pi/2,
              0: (ck[:N-4] + a)*np.pi/2,
              2: -(a+b*ck[4:])*np.pi/2,
              4: b[:-2]*np.pi/2}
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
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
            self.scale_array(c)

        elif format == 'cython' and v.ndim == 3:
            cython.Matvec.BBD_matvec3D_ptr(v, c, self[-2], self[0], self[2], self[4], axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.BBD_matvec2D_ptr(v, c, self[-2], self[0], self[2], self[4], axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.BBD_matvec1D_ptr(v, c, self[-2], self[0], self[2], self[4])
            self.scale_array(c)
        else:
            c = super(BSBSDmat, self).matvec(v, c, format=format, axis=axis)

        return c

class BCNCNmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        B_{kj} = (\phi_j, \phi_k)_w

    where

    .. math::

        j = 1, 2, ..., N-2 \text{ and } k = 1, 2, ..., N-2

    and :math:`\phi_k` is the  Mikael Neumann basis function.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], CN)
        assert isinstance(trial[0], CN)
        assert trial[0].quad == 'GC', 'Not implemented for GL'
        N = test[0].N
        M = trial[0].N
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
        self.solve = generic_PDMA(self)

class BSDHHmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        B_{kj}=(\phi_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\phi_j` is a Heinrichs Dirichlet basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], HH)
        from shenfun.la import PDMA
        ck = get_ck(test[0].N, test[0].quad)
        cp = np.ones(test[0].N); cp[2] = 2
        dk = np.ones(test[0].N); dk[:2] = 0
        if not trial[0].is_scaled():
            d = {-2: -(np.pi/8)*ck[:-4],
                  0: (np.pi/8)*(ck[:-2]*(ck[:-2]+1)+dk[:-2]),
                  2: -(np.pi/8)*(ck[:-4]+2),
                  4: (np.pi/8)}

        else:
            k = np.arange(test[0].N)
            d = {-2: -(np.pi/8)*ck[:-4]/k[1:-3],
                  0: (np.pi/8)*(ck[:-2]*(ck[:-2]+1)+dk[:-2])/k[1:-1],
                  2: -(np.pi/8)*(ck[:-4]+2)/(k[:-4]+3),
                  4: (np.pi/8)/(k[:-6]+5)}

        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
        self.solve = PDMA(self)

class CSDSNmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (\psi'_j, \phi_k)_w

    where

    .. math::

        j = 1, 2, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\phi_k` is the Shen Dirichlet basis function and :math:`\psi_j`
    the Shen Neumann basis function.

    For simplicity, the matrix is stored including the zero index row (k=0)

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SN)
        N = test[0].N
        k = np.arange(N-2, dtype=float)
        d = {-1: -((k[1:]-1)/(k[1:]+1))**2*(k[1:]+1)*np.pi,
              1: (k[:-1]+1)*np.pi}
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
        self._matvec_methods += ['cython']

    def matvec(self, v, c, format='cython', axis=0):
        c.fill(0)
        if format == 'cython' and v.ndim == 3:
            cython.Matvec.CDN_matvec3D_ptr(v, c, self[-1], self[1], axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.CDN_matvec2D_ptr(v, c, self[-1], self[1], axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.CDN_matvec1D_ptr(v, c, self[-1], self[1])
            self.scale_array(c)
        else:
            c = super(CSDSNmat, self).matvec(v, c, format=format, axis=axis)

        return c


class CSDSDmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (\phi'_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\phi_k` is the Shen Dirichlet basis function.
    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = {-1: -(k[1:N-2]+1)*np.pi,
              1: (k[:(N-3)]+1)*np.pi}
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
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
            self.scale_array(c)

        elif format == 'cython' and v.ndim == 3:
            cython.Matvec.CDD_matvec3D_ptr(v, c, self[-1], self[1], axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.CDD_matvec2D_ptr(v, c, self[-1], self[1], axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.CDD_matvec1D_ptr(v, c, self[-1], self[1])
            self.scale_array(c)
        else:
            c = super(CSDSDmat, self).matvec(v, c, format=format, axis=axis)

        return c


class CSNSDmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (\phi'_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 1, 2, ..., N-2

    and :math:`\phi_j` is the Shen Dirichlet basis function and :math:`\psi_k`
    the Shen Neumann basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N-2, dtype=float)
        def _getkey(i):
            return -(1-k[:-i]**2/(k[:-i]+2)**2)*2*np.pi
        d = dict.fromkeys(np.arange(-1, N-1, 2), _getkey)
        d[-1] = -(k[1:]+1)*np.pi
        d[1] = -(2-k[:-1]**2/(k[:-1]+2)**2*(k[:-1]+3))*np.pi
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)

    def matvec(self, v, c, format='csr', axis=0):
        c = super(CSNSDmat, self).matvec(v, c, format=format, axis=axis)
        s = [slice(None),]*v.ndim
        if self.testfunction[0].use_fixed_gauge:
            s[axis] = 0
            c[tuple(s)] = 0
        return c


class CTSDmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (\phi'_j, T_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N

    :math:`\phi_j` is the Shen Dirichlet basis function and :math:`T_k` is the
    Chebyshev basis function.
    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], T)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = dict.fromkeys(np.arange(-1, N-2, 2), -2*np.pi)
        d[-1] = -(k[1:N-1]+1)*np.pi
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)


class CSDTmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (T'_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N-2

    :math:`\phi_k` is the Shen Dirichlet basis function and :math:`T_j` is the
    Chebyshev basis function.
    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], T)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = {1: np.pi*(k[:N-2]+1)}
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)

class CTTmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (T'_j, T_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N

    :math:`T_k` is the Chebyshev basis function.
    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], T)
        assert isinstance(trial[0], T)
        N = test[0].N
        k = np.arange(N, dtype=float)
        def _getkey(i):
            return np.pi*k[i:]
        d = dict.fromkeys(np.arange(1, N, 2), _getkey)
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
        self._matvec_methods += ['cython']

    def matvec(self, v, c, format='cython', axis=0):
        c.fill(0)
        if format == 'cython' and v.ndim == 3:
            cython.Matvec.CTT_matvec3D_ptr(v, c, axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.CTT_matvec2D_ptr(v, c, axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.CTT_matvec(v, c)
            self.scale_array(c)
        else:
            c = super(CTTmat, self).matvec(v, c, format=format, axis=axis)
        return c


class CSBSDmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (\phi'_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-4

    :math:`\phi_j` is the Shen Dirichlet basis and :math:`\psi_k` the Shen
    Biharmonic basis function.
    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = {-1: -(k[1:N-4]+1)*np.pi,
              1: 2*(k[:N-4]+1)*np.pi,
              3: -(k[:N-5]+1)*np.pi}
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
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
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 3:
            cython.Matvec.CBD_matvec3D_ptr(v, c, self[-1], self[1], self[3], axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.CBD_matvec2D_ptr(v, c, self[-1], self[1], self[3], axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.CBD_matvec(v, c, self[-1], self[1], self[3])
            self.scale_array(c)
        else:
            c = super(CSBSDmat, self).matvec(v, c, format=format, axis=axis)
        return c


class CSDSBmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (\psi'_j, \phi_k)_w

    where

        j = 0, 1, ..., N-4 \text{ and } k = 0, 1, ..., N-2

    :math:`\phi_k` is the Shen Dirichlet basis function and :math:`\psi_j` the
    Shen Biharmonic basis function.
    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SB)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = {-3: (k[3:-2]-2)*(k[3:-2]+1)/k[3:-2]*np.pi,
             -1: -2*(k[1:-3]+1)**2/(k[1:-3]+2)*np.pi,
              1: (k[:-5]+1)*np.pi}
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
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
            self.scale_array(c)

        elif format == 'cython' and v.ndim == 3:
            cython.Matvec.CDB_matvec3D_ptr(v, c, self[-3], self[-1], self[1], axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.CDB_matvec2D_ptr(v, c, self[-3], self[-1], self[1], axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.CDB_matvec(v, c, self[-3], self[-1], self[1])
            self.scale_array(c)
        else:
            c = super(CSDSBmat, self).matvec(v, c, format=format, axis=axis)

        return c


class ASBSBmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\psi''_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-4 \text{ and } k = 0, 1, ..., N-4

    and :math:`\psi_k` is the Shen Biharmonic basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        N = test[0].N
        k = np.arange(N-4, dtype=float)
        d = {-2: 2*(k[2:]-1)*(k[2:]+2)*np.pi,
              0: -4*((k+1)*(k+2)**2)/(k+3)*np.pi,
              2: 2*(k[:-2]+1)*(k[:-2]+2)*np.pi}
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
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
            self.scale_array(c)

        elif format == 'cython' and v.ndim == 3:
            cython.Matvec.Tridiagonal_matvec3D_ptr(v, c, self[-2], self[0], self[2], axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.Tridiagonal_matvec2D_ptr(v, c, self[-2], self[0], self[2], axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.Tridiagonal_matvec(v, c, self[-2], self[0], self[2])
            self.scale_array(c)

        else:
            c = super(ASBSBmat, self).matvec(v, c, format=format, axis=axis)

        return c


class ASDSDmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\psi''_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Dirichlet basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=float)
        # use generator to save memory. Note that items are constant on a row for
        # keys > 2, which is taken advantage of in optimized matvecs and solvers.
        # These optimized versions never look up the diagonals for key > 2.
        def _getkey(i):
            return -4*np.pi*(k[:-(i+2)]+1)

        d = dict.fromkeys(np.arange(0, N-2, 2), _getkey)
        d[0] = -2*np.pi*(k[:N-2]+1)*(k[:N-2]+2)
        d[2] = -4*np.pi*(k[:-4]+1)
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
        self._matvec_methods += ['cython']

    def matvec(self, v, c, format='cython', axis=0):
        c.fill(0)
        if format == 'cython' and v.ndim == 3:
            cython.Matvec.ADD_matvec3D_ptr(v, c, self[0], axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.ADD_matvec2D_ptr(v, c, self[0], axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.ADD_matvec(v, c, self[0])
            self.scale_array(c)
        else:
            c = super(ASDSDmat, self).matvec(v, c, format=format, axis=axis)

        return c

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
            if u is not b:
                b = np.moveaxis(b, axis, 0)

        bs = b[s]
        us = u[s]
        if len(b.shape) == 1:
            se = 0.0
            so = 0.0
        else:
            se = np.zeros(us.shape[1:])
            so = np.zeros(us.shape[1:])

        d = self[0]
        d1 = self[2]
        M = us.shape
        us[-1] = bs[-1] / d[-1]
        us[-2] = bs[-2] / d[-2]
        for k in range(M[0]-3, -1, -1):
            if k%2 == 0:
                se += us[k+2]
                us[k] = bs[k] - d1[k]*se
            else:
                so += us[k+2]
                us[k] = bs[k] - d1[k]*so
            us[k] /= d[k]

        u /= self.scale
        self.testfunction[0].bc.set_boundary_dofs(u, True)
        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if u is not b:
                b = np.moveaxis(b, 0, axis)
        return u


class ASDSDmatW(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\psi''_j, \psi_k (x^2-1))_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Dirichlet basis function.
    w = 1-x**2

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
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\phi''_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\phi_k` is the Shen Neumann basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SN)
        N = test[0].N
        k = np.arange(N-2, dtype=float)
        def _getkey(i):
            return -4*np.pi*(k[:-i]+i)**2*(k[:-i]+1)/(k[:-i]+2)**2
        d = dict.fromkeys(np.arange(0, N-2, 2), _getkey)
        d[0] = -2*np.pi*k**2*(k+1)/(k+2)
        d[2] = -4*np.pi*(k[:-2]+2)**2*(k[:-2]+1)/(k[:-2]+2)**2
        d[4] = -4*np.pi*(k[:-4]+4)**2*(k[:-4]+1)/(k[:-4]+2)**2
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
        self._matvec_methods += ['numba']

    def matvec(self, v, c, format='csr', axis=0):
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
        d0 = -2*np.pi*(k+1)/(k+2)
        d2 = -4*np.pi*(k[:-2]+1)/(k[:-2]+2)**2
        c[N-1] = d0[N-1]*vc[N-1]
        c[N-2] = d0[N-2]*vc[N-2]
        s0 = 0
        s1 = 0
        for k in range(N-3, 0, -1):
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

    def solve(self, b, u=None, axis=0):
        assert self.shape[0] + 2 == b.shape[0]
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
        j2 = np.arange(self.shape[0])**2
        j2[0] = 1
        j2 = 1./j2
        if len(b.shape) == 1:
            se = 0.0
            so = 0.0
        else:
            se = np.zeros(u.shape[1:])
            so = np.zeros(u.shape[1:])
            j2.repeat(np.prod(bs.shape[1:])).reshape(bs.shape)
        d = self[0]*j2
        d1 = self[2]*j2[2:]

        M = us.shape
        us[-1] = bs[-1] / d[-1]
        us[-2] = bs[-2] / d[-2]
        for k in range(M[0]-3, 0, -1):
            if k%2 == 0:
                se += us[k+2]
                us[k] = bs[k] - d1[k]*se
            else:
                so += us[k+2]
                us[k] = bs[k] - d1[k]*so
            us[k] /= d[k]
        sl = [np.newaxis]*b.ndim
        sl[0] = slice(None)
        us *= j2[tuple(sl)]

        u /= self.scale
        if self.testfunction[0].use_fixed_gauge:
            u[0] = self.testfunction[0].mean/(np.pi/self.testfunction[0].domain_factor())
        self.testfunction[0].bc.set_boundary_dofs(u, True)
        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if u is not b:
                b = np.moveaxis(b, 0, axis)
        return u


class ATTmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\psi''_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N

    and :math:`\psi_k` is the Chebyshev basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], T)
        assert isinstance(trial[0], T)
        N = test[0].N
        k = np.arange(N, dtype=float)
        def _getkey(j):
            return k[j:]*(k[j:]**2-k[:-j]**2)*np.pi/2.
        d = dict.fromkeys(np.arange(2, N, 2), _getkey)
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
        self._matvec_methods += ['cython']

    def matvec(self, v, c, format='cython', axis=0):
        c.fill(0)
        if format == 'cython' and v.ndim == 3:
            cython.Matvec.ATT_matvec3D_ptr(v, c, axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.ATT_matvec2D_ptr(v, c, axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.ATT_matvec(v, c)
            self.scale_array(c)
        else:
            c = super(ATTmat, self).matvec(v, c, format=format, axis=axis)
        return c

class ACNCNmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\phi''_j, \phi_k)_w

    where

    .. math::

        j = 1, 2, ..., N-2 \text{ and } k = 1, 2, ..., N-2

    and :math:`\phi_k` is the combined Shen Neumann basis function.

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
        self.solve = generic_TDMA(self)

class ATUSDmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\phi''_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, 2, ..., N-2 \text{ and } k = 0, 1, 2, ..., N-2

    and :math:`\phi_k` is Shen's Dirichlet basis and :math:`\psi_j`
    is the MixedTU basis. This is a Petrov-Galerkin matrix.
    w = (1-x**2)

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], TU)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=float)
        if test[0].is_scaled():
            d = {0: -0.5*np.pi*k[1:-1]*k[2:]}
        else:
            d = {0: -0.5*np.pi*k[1:-1]*k[2:]**2*(k[2:]+1)}
        d[2] = -d[0][:-2].copy()
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self.solver = TwoDMA(self)

class ADUSDmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\phi''_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, 2, ..., N-2 \text{ and } k = 0, 1, 2, ..., N-2

    and :math:`\phi_k` is Shen's Dirichlet basis and :math:`\psi_j`
    is the DirichletU basis. This is a Petrov-Galerkin matrix.
    w = -(1-x**2)

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], DU)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = {0: -np.pi*k[1:-1]*k[2:]}
        d[2] = -d[0][:-2].copy()
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self.solver = TwoDMA(self)

#class ASDHHmat(SpectralMatrix):
#    r"""Stiffness matrix for inner product
#
#    .. math::
#
#        A_{kj} = (\phi''_j, \psi_k)_w
#
#    where
#
#    .. math::
#
#        j = 0, 1, 2, ..., N-2 \text{ and } k = 0, 1, 2, ..., N-2
#
#    and :math:`\phi_k` is Shen's Dirichlet basis and :math:`\psi_j`
#    is the Heinrichs basis.
#
#    """
#    def __init__(self, test, trial, scale=1, measure=1):
#        assert isinstance(test[0], SD)
#        assert isinstance(trial[0], HH)
#        N = test[0].N
#        k = np.arange(N, dtype=float)
#        ck = get_ck(N, test[0].quad)
#        d = {0: -np.pi/2*ck[:-2]*k[1:-1]*k[2:],
#             2: np.pi/2*k[:-4]*k[1:-3]}
#        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
#        self.solver = TwoDMA(self)

class ASDHHmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\phi''_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, 2, ..., N-2 \text{ and } k = 0, 1, 2, ..., N-2

    and :math:`\phi_k` is Shen's Dirichlet basis and :math:`\psi_j`
    is the Heinrichs basis.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], HH)
        N = test[0].N
        k = np.arange(N, dtype=float)
        ck = get_ck(N, test[0].quad)

        if trial[0].is_scaled():
            d = {0: -np.pi/2*ck[:-2]*k[2:],
                 2: np.pi/2*k[:-4]*k[1:-3]/(k[:-4]+3)}
        else:
            d = {0: -np.pi/2*ck[:-2]*k[2:]*k[1:-1],
                 2: np.pi/2*k[:-4]*k[1:-3]}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self.solver = TwoDMA(self)

class ATUSNmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\phi''_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, 2, ..., N-2 \text{ and } k = 0, 1, 2, ..., N-2

    and :math:`\phi_k` is Shen's Neumann basis and :math:`\psi_j`
    is the MixedTU basis. This is a Petrov-Galerkin matrix.
    w = (1-x**2).

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], TU)
        assert isinstance(trial[0], SN)
        N = test[0].N
        k = np.arange(N, dtype=float)
        if test[0].is_scaled():
            d = {0: -(np.pi/2)*k[:-2]**2*(k[1:-1]/k[2:]),
                 2: (np.pi/2)*k[1:-3]*k[2:-2]}
        else:
            d = {0: -(np.pi/2)*k[:-2]**2*k[1:-1]*(k[:-2]+3),
                 2: (np.pi/2)*k[1:-3]*k[2:-2]**2*(k[2:-2]+1)}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self.solver = TwoDMA(self)

class ADUSNmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\phi''_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, 2, ..., N-2 \text{ and } k = 0, 1, 2, ..., N-2

    and :math:`\phi_k` is Shen's Neumann basis and :math:`\psi_j`
    is the DirichletU basis. This is a Petrov-Galerkin matrix.
    w = (1-x**2).

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], DU)
        assert isinstance(trial[0], SN)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = {0: -np.pi*k[:-2]**2*(k[1:-1]/k[2:]),
             2: np.pi*k[1:-3]*k[2:-2]}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self.solver = TwoDMA(self)

class AHHHHmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\psi''_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Heinrichs Dirichlet basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], HH)
        assert isinstance(trial[0], HH)
        N = test[0].N
        k = np.arange(N-2, dtype=float)
        ck = get_ck(N-2, test[0].quad)
        dk = ck.copy()
        dk[:2] = 0
        if not test[0].is_scaled():
            d = {0: -np.pi/16*ck**2*2*((k+1)*(k+2)+dk*(k-2)*(k-1)),
                 2: np.pi/16*ck[:-2]*2*k[:-2]*k[1:-1],
                 -2: np.pi/16*ck[:-2]*2*k[2:]*k[1:-1]}
        else:
            d = {0: -np.pi/8*ck**2*((k+2)/(k+1)+dk*(k-2)*(k-1)/(k+1)**2),
                 2: np.pi/8*ck[:-2]*k[:-2]/(k[:-2]+3),
                 -2: np.pi/8*ck[:-2]*k[2:]/(k[1:-1]+2)}
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)


class BTUSDmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        B_{kj} = (\phi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\phi_k` is Shen's Dirichlet basis and :math:`\psi_j`
    is the MixedTU basis. This is a Petrov-Galerkin matrix.
    w = (1-x**2).

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], TU)
        assert isinstance(trial[0], SD)
        N = test[0].N
        ck = get_ck(N, test[0].quad)
        k = np.arange(N, dtype=float)
        d = {-2: -(np.pi/8),
              0: (np.pi/8)*ck[:-2]+(np.pi/4)*(k[:-2]+2)/(k[:-2]+3),
              2: -(np.pi/8)*(k[:-4]+1)/(k[:-4]+3)-(np.pi/4)*(k[:-4]+2)/(k[:-4]+3),
              4: (np.pi/8)*(k[:-6]+1)/(k[:-6]+3)}
        if not test[0].is_scaled():
            d[-2] *= (k[2:-2]+2)*(k[2:-2]+3)
            d[0] *= (k[:-2]+2)*(k[:-2]+3)
            d[2] *= (k[:-4]+2)*(k[:-4]+3)
            d[4] *= (k[:-6]+2)*(k[:-6]+3)

        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self.solver = FDMA(self)

class BDUSDmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        B_{kj} = (\phi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\phi_k` is Shen's Dirichlet basis and :math:`\psi_j`
    is the DirichletU basis. This is a Petrov-Galerkin matrix.
    w = -(1-x**2).

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], DU)
        assert isinstance(trial[0], SD)
        N = test[0].N
        ck = get_ck(N, test[0].quad)
        k = np.arange(N, dtype=float)
        d = {-2: -(np.pi/4),
              0: (np.pi/4)*(ck[:-2]+2*(k[:-2]+2)/(k[:-2]+3)),
              2: -(np.pi/4)*((k[:-4]+1)/(k[:-4]+3)+2*(k[:-4]+2)/(k[:-4]+3)),
              4: (np.pi/4)*(k[:-6]+1)/(k[:-6]+3)}

        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self.solver = FDMA(self)

class BHHHHmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        B_{kj}=(\phi_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\phi_j` is a Heinrichs Dirichlet basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], HH)
        assert isinstance(trial[0], HH)
        from shenfun.la import PDMA
        ck = get_ck(test[0].N-2, test[0].quad)
        cp = np.ones(test[0].N-2); cp[2] = 2
        dk = np.ones(test[0].N-2); dk[:2] = 0
        if not test[0].is_scaled():
            d = {0: np.pi/16*(ck**2*(ck+1)/2 + dk*(cp+3)/2),
                 2: -np.pi/16*(ck[:-2]+ck[:-2]**2/2+dk[:-2]/2),
                 4: np.pi*ck[:-4]/32}

        else:
            k = np.arange(test[0].N)
            d = {0: np.pi/16*(ck**2*(ck+1)/2 + dk*(cp+3)/2)/k[1:-1]**2,
                 2: -np.pi/16*(ck[:-2]+ck[:-2]**2/2+dk[:-2]/2)/k[1:-3]/k[3:-1],
                 4: np.pi*ck[:-4]/32/k[1:-5]/k[5:-1]}
        d[-2] = d[2].copy()
        d[-4] = d[4].copy()
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
        self.solve = PDMA(self)

class BSDMNmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        B_{kj} = (\phi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\phi_k` is Mikael's Neumann basis and :math:`\psi_j`
    is Shen's Dirichlet basis. This is a Petrov-Galerkin matrix.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], MN)
        N = test[0].N
        k = np.arange(N, dtype=float)
        kp = k.copy(); kp[0] = 1
        qk = np.ones(N)
        qk[0] = 0
        qk[1] = 1.5
        dk = np.ones(N)
        dk[0] = 0
        d = {-2: -np.pi/2/k[2:-2],
              0: (np.pi/2)*(2*qk[:-2]/kp[:-2] + 1/k[2:]),
              2: -(np.pi/2)*(1/kp[:-4] + 2/k[2:-2]),
              4: (np.pi/2)/k[2:-4]}
        d[-2][0] = 0
        d[0][0] = np.pi
        d[2][0] = -np.pi/2
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

class BSNCNmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        B_{kj} = (\phi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\phi_k` is Mikael's Neumann basis and :math:`\psi_j`
    is Shen's Neumann basis.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], CN)
        N = test[0].N
        k = np.arange(N, dtype=float)
        kp = k.copy(); kp[0] = 1
        qk = np.ones(N)
        qk[0] = 0
        qk[1] = 1.5
        dk = np.ones(N)
        dk[:3] = 0
        d = {-2: -np.pi/2/k[2:-2]**2,
              0: (np.pi/2)*((1+dk[:-2])/kp[:-2]**2+ (k[:-2]/k[2:])**2/k[2:]**2),
              2: -(np.pi/2)*(1/kp[:-4]**2+2*(k[:-4]/k[2:-2])**2/k[2:-2]**2),
              4: (np.pi/2)*(k[:-6]/k[2:-4])**2/k[2:-4]**2}
        d[-2][0] = 0
        d[0][0] = np.pi
        d[2][0] = 0
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self.solver = FDMA(self)

class ASDMNmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        B_{kj} = (\phi^{''}_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\phi_k` is Mikael's Neumann basis and :math:`\psi_j`
    is Shen's Dirichlet basis. This is a Petrov-Galerkin matrix.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], MN)
        N = test[0].N
        k = np.arange(N, dtype=float)
        qk = np.ones(N)
        qk[0] = 0
        qk[1] = 1.5
        dk = np.ones(N)
        dk[0] = 0
        d = {0: -2*np.pi*k[1:-1],
             2: 2*np.pi*k[1:-3]}
        d[0][0] = 0
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self.solver = TwoDMA(self)

class ASNCNmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        B_{kj} = (\phi^{''}_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\phi_k` is Mikael's Neumann basis and :math:`\psi_j`
    is Shen's Neumann basis. This is a Petrov-Galerkin matrix.

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
        self.solver = TwoDMA(self)

class BTUSNmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        B_{kj} = (\phi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, 2, ..., N-2 \text{ and } k = 0, 1, 2, ..., N-2

    and :math:`\phi_k` is Shen's Dirichlet basis and :math:`\psi_j`
    is the MixedTU basis. This is a Petrov-Galerkin matrix.
    w = (1-x**2).

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], TU)
        assert isinstance(trial[0], SN)
        N = test[0].N
        ck = get_ck(N, test[0].quad)
        k = np.arange(N, dtype=float)
        d = {-2: -(np.pi/8)*(k[2:-2]-2)**2/k[2:-2]**2,
              0: (np.pi/8)*(ck[:-2]+2*(k[:-2]/(k[:-2]+3))*(k[:-2]/(k[:-2]+2))),
              2: -(np.pi/8)*(2*k[2:-2]/k[3:-1]+k[1:-3]/k[3:-1]*(k[2:-2]/k[4:])**2),
              4: (np.pi/8)*k[1:-5]/k[3:-3]}
        if not test[0].is_scaled():
            d[-2] *= k[4:]*(k[4:]+1)
            d[0] *= k[2:]*(k[2:]+1)
            d[2] *= k[2:-2]*k[3:-1]
            d[4] *= k[2:-4]*k[3:-3]
        d[-2][0] = 0
        d[0][0] = np.pi
        d[2][0] = 0
        d[4][0] = 0

        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self.solver = FDMA(self)

class BDUSNmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        B_{kj} = (\phi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, 2, ..., N-2 \text{ and } k = 0, 1, 2, ..., N-2

    and :math:`\phi_k` is Shen's Dirichlet basis and :math:`\psi_j`
    is the DirichletU basis. This is a Petrov-Galerkin matrix.
    w = (1-x**2).

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], DU)
        assert isinstance(trial[0], SN)
        N = test[0].N
        ck = get_ck(N, test[0].quad)
        k = np.arange(N, dtype=float)
        d = {-2: -(np.pi/4)*(k[2:-2]-2)**2/k[2:-2]**2,
              0: (np.pi/4)*(ck[:-2]+2*(k[:-2]/(k[:-2]+3))*(k[:-2]/(k[:-2]+2))),
              2: -(np.pi/4)*(2*k[2:-2]/k[3:-1]+k[1:-3]/k[3:-1]*(k[2:-2]/k[4:])**2),
              4: (np.pi/4)*k[1:-5]/k[3:-3]}

        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)
        self.solver = FDMA(self)

class SSBSBmat(SpectralMatrix):
    r"""Biharmonic matrix for inner product

    .. math::

        S_{kj} = (\psi''''_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-4 \text{ and } k = 0, 1, ..., N-4

    and :math:`\psi_k` is the Shen Biharmonic basis function.
    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        N = test[0].N
        ki = np.arange(N-4)
        k = np.arange(N-4, dtype=float)
        def _getkey(j):
            i = 8*(ki[:-j]+1)*(ki[:-j]+2)*(ki[:-j]*(ki[:-j]+4)+3*(ki[j:]+2)**2)
            return np.array(i*np.pi/(k[j:]+3))
        d = dict.fromkeys(np.arange(0, N-4, 2), _getkey)
        d[0] = np.pi*8*(ki+1)**2*(ki+2)*(ki+4)
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
        self._matvec_methods += ['cython']

    def matvec(self, v, c, format='cython', axis=0):
        c.fill(0)
        if format == 'cython' and v.ndim == 3:
            cython.Matvec.SBB_matvec3D_ptr(v, c, self[0], axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.SBB_matvec2D_ptr(v, c, self[0], axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.SBBmat_matvec(v, c, self[0])
            self.scale_array(c)

        else:
            c = super(SSBSBmat, self).matvec(v, c, format=format, axis=axis)

        return c


class BSDBCDmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\psi_j, \phi_k)_w

    where

    .. math::

        j = 0, 1 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_j` is the Dirichlet boundary basis and
    :math:`\phi_k` is the Shen Dirichlet basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], BCD)
        d = {0: np.array([np.pi/2, np.pi/4]),
             1: np.array([np.pi/2]),
            -1: np.array([-np.pi/4, 0])}

        SpectralMatrix.__init__(self, d, test, trial, measure=measure)


class BSBBCBmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\psi_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, 2, 3 \text{ and } k = 0, 1, ..., N-4

    and :math:`\psi_j` is the Biharmonic boundary basis and
    :math:`\phi_k` is the Shen Biharmonic basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], BCB)
        d = {0: np.array([np.pi/2, np.pi*21/64, -np.pi/16, np.pi/32]),
             1: np.array([np.pi/2, -np.pi*5/64, np.pi/16]),
             2: np.array([np.pi*5/24, -np.pi*5/64]),
             3: np.array([-np.pi*5/24]),
            -1: np.array([-np.pi*21/64, 0, np.pi/32, 0]),
            -2: np.array([0, -np.pi/32, 0, 0]),
            -3: np.array([np.pi/32, 0, 0, 0])}
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)


class _Chebmatrix(SpectralMatrix):
    def __init__(self, test, trial, measure=1):
        SpectralMatrix.__init__(self, {}, test, trial, measure=measure)


class _ChebMatDict(dict):
    """Dictionary of inner product matrices

    Matrices that are missing keys are generated from Vandermonde type
    computations.

    """

    def __missing__(self, key):
        measure = 1 if len(key) == 2 else key[3]
        c = functools.partial(_Chebmatrix, measure=measure)
        self[key] = c
        return c

    def __getitem__(self, key):
        matrix = dict.__getitem__(self, key)
        #assert key[0][1] == 0, 'Test cannot be differentiated (weighted space)'
        return matrix


# Define dictionary to hold all predefined matrices
# When looked up, missing matrices will be generated automatically
mat = _ChebMatDict({
    ((T,  0), (T , 0)): BTTmat,
    ((SD, 0), (SD, 0)): BSDSDmat,
    ((SD, 0), (SD, 0), (-1, 1), (x**2-1)): functools.partial(BSDSDmatW, scale=-1, measure=(x**2-1)),
    ((SD, 0), (SD, 0), (-1, 1), (1-x**2)): functools.partial(BSDSDmatW, measure=(1-x**2)),
    ((SB, 0), (SB, 0)): BSBSBmat,
    ((SN, 0), (SN, 0)): BSNSNmat,
    ((CN, 0), (CN, 0)): BCNCNmat,
    ((SN, 0), (T , 0)): BSNTmat,
    ((SN, 0), (SB, 0)): BSNSBmat,
    ((SD, 0), (SN, 0)): BSDSNmat,
    ((SN, 0), (SD, 0)): BSNSDmat,
    ((T , 0), (SN, 0)): BTSNmat,
    ((SB, 0), (SD, 0)): BSBSDmat,
    ((T,  0), (SD, 0)): BTSDmat,
    ((SD, 0), (T,  0)): BSDTmat,
    ((SD, 0), (SD, 2)): ASDSDmat,
    ((SD, 0), (SD, 2), (-1, 1), (x**2-1)): functools.partial(ASDSDmatW, scale=-1, measure=(x**2-1)),
    ((SD, 0), (SD, 2), (-1, 1), (1-x**2)): functools.partial(ASDSDmatW, measure=(1-x**2)),
    ((T , 0), (T , 2)): ATTmat,
    ((SN, 0), (SN, 2)): ASNSNmat,
    ((CN, 0), (CN, 2)): ACNCNmat,
    ((SB, 0), (SB, 2)): ASBSBmat,
    ((HH, 0), (HH, 2)): AHHHHmat,
    ((HH, 0), (HH, 0)): BHHHHmat,
    ((SD, 0), (MN, 0)): BSDMNmat,
    ((SD, 0), (MN, 2)): ASDMNmat,
    ((SN, 0), (CN, 2)): ASNCNmat,
    ((SN, 0), (CN, 0)): BSNCNmat,
    ((SD, 0), (HH, 0)): BSDHHmat,
    ((SD, 0), (HH, 2)): ASDHHmat,
    ((DU, 0), (SD, 2), (-1, 1), (1-x**2)): functools.partial(ADUSDmat, measure=(1-x**2)),
    ((DU, 0), (SD, 0), (-1, 1), (1-x**2)): functools.partial(BDUSDmat, measure=(1-x**2)),
    ((TU, 0), (SD, 2), (-1, 1), (1-x**2)): functools.partial(ATUSDmat, measure=(1-x**2)),
    ((TU, 0), (SD, 0), (-1, 1), (1-x**2)): functools.partial(BTUSDmat, measure=(1-x**2)),
    ((TU, 0), (SN, 2), (-1, 1), (1-x**2)): functools.partial(ATUSNmat, measure=(1-x**2)),
    ((TU, 0), (SN, 0), (-1, 1), (1-x**2)): functools.partial(BTUSNmat, measure=(1-x**2)),
    ((DU, 0), (SN, 2), (-1, 1), (1-x**2)): functools.partial(ADUSNmat, measure=(1-x**2)),
    ((DU, 0), (SN, 0), (-1, 1), (1-x**2)): functools.partial(BDUSNmat, measure=(1-x**2)),
    ((SB, 0), (SB, 4)): SSBSBmat,
    ((SD, 0), (SN, 1)): CSDSNmat,
    ((SB, 0), (SD, 1)): CSBSDmat,
    ((T,  0), (SD, 1)): CTSDmat,
    ((T,  0), (T,  1)): CTTmat,
    ((SD, 0), (SD, 1)): CSDSDmat,
    ((SN, 0), (SD, 1)): CSNSDmat,
    ((SD, 0), (SB, 1)): CSDSBmat,
    ((SD, 0), (T,  1)): CSDTmat,
    ((SD, 0), (BCD, 0)): BSDBCDmat,
    ((SB, 0), (BCB, 0)): BSBBCBmat,
    })
