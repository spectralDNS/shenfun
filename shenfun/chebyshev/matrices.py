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
letters in the matrix name uses the 'short_name' method for all these different
bases, see chebyshev.bases.py.

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

Check that this matrix corresponds to the matrix 'd' hardcoded below:

>>> import numpy as np
>>> d = {-2: -np.pi/2,
...       0: np.array([ 1.5*np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]),
...       2: -np.pi/2}
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

import numpy as np
import scipy
from shenfun.optimization import cython, numba
from shenfun.matrixbase import SpectralMatrix, SpectralMatDict
from shenfun.spectralbase import get_norm_sq
from shenfun.la import TDMA as generic_TDMA
from shenfun.la import PDMA as generic_PDMA
from shenfun.la import TwoDMA
from shenfun.legendre import bases as legendrebases
from .la import ADDSolver, ANNSolver
from . import bases


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
L = legendrebases.Orthogonal

BCG = bases.BCGeneric

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
    def __init__(self, test, trial, scale=1, measure=1, assemble=None, kind=None, fixed_resolution=None):
        assert isinstance(test[0], T)
        assert isinstance(trial[0], T)
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
        self._matvec_methods += ['self']

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        return {0: get_norm_sq(test[0], trial[0], method)}

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

    def solve(self, b, u=None, axis=0, constraints=()):
        """Solve matrix system Au = b

        where A is the current matrix (self)

        Parameters
        ----------
        b : array
            Array of right hand side on entry and solution on exit is u is None
        u : array, optional
            Solution array if provided
        axis : int, optional
            The axis over which to solve for if u is multi-
            dimensional
        constraints : tuple of 2-tuples
            The 2-tuples represent (row, val)
            The constraint indents the matrix row and sets b[row] = val

        """
        assert constraints == ()
        if u is None:
            u = b
        else:
            u[:] = b
        space = self.testfunction[0]
        u *= (2/np.pi*space.domain_factor())
        u[space.si[0]] /= 2
        if space.quad == 'GL':
            u[space.si[-1]] /= 2
        return u

class BTLmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(L_j, T_k)_w,

    where :math:`T_k \in` :class:`.chebyshev.bases.Orthogonal`,
    :math:`L_j \in` :class:`shenfun.legendre.bases.Orthogonal`,  and test
    and trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1, assemble=None, kind=None, fixed_resolution=None):
        assert isinstance(test[0], T)
        assert isinstance(trial[0], L)
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)

    def assemble(self, method):
        gammaln = scipy.special.gammaln
        test = self.testfunction
        trial = self.trialfunction
        M = test[0].N
        N = trial[0].N
        Q = min(M, N)
        k = np.arange(Q)
        self.a = np.exp(gammaln((2*k+1)/2) - gammaln((2*k+2)/2))
        d = {}
        for n in range(0, N, 2):
            d[n] = np.exp(gammaln((n+1)/2)-gammaln((n+2)/2)) * np.exp(gammaln((2*k[:N-n]+n+1)/2) - gammaln((2*k[:N-n]+n+2)/2))
        if test[0].quad == 'GL':
            d[0][-1] *= 2
        return d

    def matvec(self, v, c, format=None, axis=0):
        # Roll relevant axis to first
        c.fill(0)
        if axis > 0:
            v = np.moveaxis(v, axis, 0)
            c = np.moveaxis(c, axis, 0)

        N = self.testfunction[0].N
        for n in range(0, N, 2):
            c[:(N-n)] += self.a[n//2]*self.a[n//2:(N-n//2)]*v[n:]

        if axis > 0:
            c = np.moveaxis(c, 0, axis)
            v = np.moveaxis(v, 0, axis)
        return c

class BSDSDmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\phi_j, \phi_k)_w,

    where :math:`\phi_j \in` :class:`.chebyshev.bases.ShenDirichlet`, and test
    and trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1, assemble=None, kind=None, fixed_resolution=None):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
        self._matvec_methods += ['cython', 'self']

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        h = get_norm_sq(test[0], trial[0], method)
        d = {
            -2: -h[1],
            0: h[:-2]+h[2:],
            2: -h[1]
        }
        return d

    def get_solver(self):
        return generic_TDMA

    def matvec(self, v, c, format=None, axis=0):
        format = 'cython' if format is None else format
        N, M = self.shape
        c.fill(0)
        # Cython implementation only handles square matrix
        if not M == N:
            format = 'csr'

        d0 = np.array(self[0]).astype(float)
        ld = np.array(self[-2]).astype(float)*np.ones(M-2)
        if format == 'cython':
            cython.Matvec.Tridiagonal_matvec(v, c, axis, ld, d0, ld)
            self.scale_array(c, self.scale)
        elif format == 'self':
            if axis > 0:
                v = np.moveaxis(v, axis, 0)
                c = np.moveaxis(c, axis, 0)
            s = (slice(0, M),)+(np.newaxis,)*(v.ndim-1) # broadcasting
            c[:(M-2)] = float(self[-2])*v[2:M]
            c[:M] += d0[s]*v[:M]
            c[2:M] += float(self[2])*v[:(M-2)]
            if axis > 0:
                v = np.moveaxis(v, 0, axis)
                c = np.moveaxis(c, 0, axis)
            self.scale_array(c, self.scale)

        else:
            format = None if format in ('cython', 'self') else format
            c = super(BSDSDmat, self).matvec(v, c, format=format, axis=axis)

        return c

class BSNSDmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenNeumann`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SD)
        h = get_norm_sq(test[0], trial[0], method)
        k = np.arange(len(h))
        alpha = k/(k+2)
        M, N = test[0].N-2, trial[0].N-2
        d = {
            0: h[:-2] + alpha[:-2]**2*h[2:],
            2: -alpha[:dmax(M, N, 2)]**2*h[2:dmax(M, N, 2)+2],
            -2: -np.pi/2
        }
        return d

class BSDSNmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenDirichlet`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.ShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1, assemble=None, kind=None, fixed_resolution=None):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SN)
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
        self._matvec_methods += ['cython']

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        h = get_norm_sq(test[0], trial[0], method)
        k = np.arange(len(h))
        alpha = k/(k+2)
        M, N = test[0].N-2, trial[0].N-2
        d = {
            0: h[:-2] + alpha[:-2]**2*h[2:],
            -2: -alpha[:dmax(M, N, -2)]**2*h[2:dmax(M, N, -2)+2],
            2: -np.pi/2
        }
        return d

    def matvec(self, v, c, format=None, axis=0):
        format = 'cython' if format is None else format
        c.fill(0)
        N, M = self.shape
        if not M == N:
            format = 'csr'
        if format == 'cython' and v.ndim == 3:
            cython.Matvec.BDN_matvec(v, c, axis, self[-2], self[0], self[2])
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
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], LD)
        assert isinstance(trial[0], LD)
        h = get_norm_sq(test[0], trial[0], method)
        d = {0: h[:-1]+h[1:],
             1: np.pi/2,
            -1: np.pi/2}
        return d

class BSNSNmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\phi_j, \phi_k)_w,

    where :math:`\phi_k \in` :class:`.chebyshev.bases.ShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SN)
        h = get_norm_sq(test[0], trial[0], method)
        M, N = test[0].N-2, trial[0].N-2
        k = np.arange(len(h))
        alpha = (k/(k+2))**2
        d = {
            0: h[:-2] + alpha[:-2]**2*h[2:],
            2: -alpha[:dmax(M, N, 2)]*h[2:(dmax(M, N, 2)+2)]
        }
        d[-2] = d[2].copy()
        return d

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
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], T)
        h = get_norm_sq(test[0], trial[0], method)
        M, N = test[0].N-2, trial[0].N
        d = {
            0: h[:min(M, N)],
            2: -h[2:(dmax(M, N, 2)+2)]
        }
        return d


class BTSDmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, T_k)_w,

    where the test function :math:`T_k \in` :class:`.chebyshev.bases.Orthogonal`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], T)
        assert isinstance(trial[0], SD)
        h = get_norm_sq(test[0], trial[0], method)
        M, N = test[0].N, trial[0].N-2
        d = {
            0: h[:min(M, N)],
            -2: -h[2:(dmax(M, N, -2)+2)]
        }
        return d

class BTSNmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, T_k)_w,

    where the test function :math:`T_k \in` :class:`.chebyshev.bases.Orthogonal`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.ShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], T)
        assert isinstance(trial[0], SN)
        h = get_norm_sq(test[0], trial[0], method)
        M, N = test[0].N, trial[0].N-2
        alpha = trial[0].stencil_matrix()[2]
        d = {
            0: h[:-2],
            -2: h[2:(dmax(M, N, -2)+2)]*alpha
        }
        return d

class BSBSBmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\phi_j, \phi_k)_w,

    where :math:`\phi_k \in` :class:`.chebyshev.bases.ShenBiharmonic`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1, assemble=None, kind=None, fixed_resolution=None):
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
        self._matvec_methods += ['cython', 'self']

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        h = get_norm_sq(test[0], trial[0], method)
        S = trial[0].stencil_matrix()
        d = {
            0: h[:-4] + h[2:-2]*S[2][:-2]**2 + h[4:]*S[4]**2,
            2: h[2:-4]*S[2][:-4] + h[4:-2]*S[4][:-2]*S[2][2:-2],
            4: h[4:-4]*S[4][:-4]
        }
        d[-2] = d[2].copy()
        d[-4] = d[4].copy()
        return d

    def get_solver(self):
        return generic_PDMA

    def matvec(self, v, c, format=None, axis=0):
        format = 'cython' if format is None else format
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
            cython.Matvec.Pentadiagonal_matvec(v, c, axis, self[-4], self[-2], self[0],
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
    def __init__(self, test, trial, scale=1, measure=1, assemble=None, kind=None, fixed_resolution=None):
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
        self._matvec_methods += ['cython', 'self']

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SD)
        h = get_norm_sq(test[0], trial[0], method)
        N = test[0].N-4
        M = trial[0].N-2
        Q = min(M, N)
        k = np.arange(Q, dtype=float)
        a = 2*(k+2)/(k+3)
        b = (k+1)/(k+3)
        d = {
            -2: -np.pi/2,
            0: (h[:Q]+a*np.pi/2),
            2: -(a[:min(Q, M-2)]*np.pi/2+h[4:min(Q+4, M+2)]*b[:min(Q, M-2)]),
            4: b[:min(Q, M-4)]*h[4:min(Q+4, M)]
        }
        return d

    def matvec(self, v, c, format=None, axis=0):
        format = 'cython' if format is None else format
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

        elif format == 'cython':
            cython.Matvec.BBD_matvec(v, c, axis, self[-2], self[0], self[2], self[4])
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
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], T)
        N, M = test[0].N-4, trial[0].N
        Q = min(M, N)
        h = get_norm_sq(test[0], trial[0], method)
        k = np.arange(Q, dtype=float)
        d = {0: h[:Q],
             2: -np.pi*(k[:min(Q, M-2)]+2)/(k[:min(Q, M-2)]+3),
             4: h[4:min(Q+4, M)]*(k[:min(Q, M-4)]+1)/(k[:min(Q, M-4)]+3)}
        return d

class CSDSNmat(SpectralMatrix):
    r"""Derivative matrix :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(\psi'_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenDirichlet`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.ShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1, assemble=None, kind=None, fixed_resolution=None):
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
        self._matvec_methods += ['cython']

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SN)
        N = test[0].N
        k = np.arange(N-2, dtype=float)
        d = {-1: -((k[1:]-1)/(k[1:]+1))**2*(k[1:]+1)*np.pi,
              1: (k[:-1]+1)*np.pi}
        return d

    def matvec(self, v, c, format=None, axis=0):
        format = 'cython' if format is None else format
        c.fill(0)
        if format == 'cython':
            c = cython.Matvec.CDN_matvec(v, c, axis, self[-1], self[1])
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
    def __init__(self, test, trial, scale=1, measure=1, assemble=None, kind=None, fixed_resolution=None):
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
        self._matvec_methods += ['cython', 'self']

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = {-1: -(k[1:N-2]+1)*np.pi,
              1: (k[:(N-3)]+1)*np.pi}
        return d

    def matvec(self, v, c, format=None, axis=0):
        format = 'cython' if format is None else format
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
        elif format == 'cython':
            c = cython.Matvec.CDD_matvec(v, c, axis, self[-1], self[1])
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
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
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
        return d

class CTSDmat(SpectralMatrix):
    r"""Derivative matrix :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(\psi'_j, T_k)_w,

    where the test function :math:`T_k \in` :class:`.chebyshev.bases.Orthogonal`, the
    trial :math:`\psi_j \in` :class:`.chebyshev.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], T)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = dict.fromkeys(np.arange(-1, N-2, 2), -2*np.pi)
        d[-1] = -(k[1:N-1]+1)*np.pi
        return d

    def matvec(self, v, c, format=None, axis=0):
        format = 'cython' if format is None else format
        c.fill(0)
        if format == 'cython':
            cython.Matvec.CTSD_matvec(v, c, axis)
            self.scale_array(c, self.scale)
        else:
            format = None if format in self._matvec_methods else format
            c = super(CTSDmat, self).matvec(v, c, format=format, axis=axis)
        return c

class ASBTmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (T''_j, \phi_k)_w

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenBiharmonic`, the trial
    function :math:`T_j \in` :class:`.chebyshev.bases.Orthogonal`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], T)
        N, M = test[0].N-4, trial[0].N
        Q = min(N, M)
        k = np.arange(Q, dtype=float)
        d = {2: 2*np.pi*(k[:min(Q, M-2)]+2)*(k[:min(Q, M-2)]+1)}
        return d


class CSDTmat(SpectralMatrix):
    r"""Derivative matrix :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(T'_j, \phi_k)_w,

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenDirichlet`, the
    trial :math:`T_j \in` :class:`.chebyshev.bases.Orthogonal`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], T)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = {1: np.pi*(k[:N-2]+1)}
        return d

class CTTmat(SpectralMatrix):
    r"""Derivative matrix :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(T'_j, T_k)_w,

    where :math:`T_j \in` :class:`.chebyshev.bases.Orthogonal`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1, assemble=None, kind=None, fixed_resolution=None):
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
        self._matvec_methods += ['cython']

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], T)
        assert isinstance(trial[0], T)
        M, N = test[0].N, trial[0].N
        k = np.arange(min(M, N), dtype=float)
        self._keyscale = 1
        def _getkey(i):
            return np.pi*self._keyscale*k[i:]
        d = dict.fromkeys(np.arange(1, N, 2), _getkey)
        return d

    def matvec(self, v, c, format=None, axis=0):
        format = 'cython' if format is None else format
        c.fill(0)
        if format == 'cython':
            cython.Matvec.CTT_matvec(v, c, axis)
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
    def __init__(self, test, trial, scale=1, measure=1, assemble=None, kind=None, fixed_resolution=None):
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
        self._matvec_methods += ['cython', 'self']

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = {-1: -(k[1:N-4]+1)*np.pi,
              1: 2*(k[:N-4]+1)*np.pi,
              3: -(k[:N-5]+1)*np.pi}
        return d

    def matvec(self, v, c, format=None, axis=0):
        format = 'cython' if format is None else format
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
        elif format == 'cython':
            cython.Matvec.CBD_matvec(v, c, axis, self[-1], self[1], self[3])
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
    def __init__(self, test, trial, scale=1, measure=1, assemble=None, kind=None, fixed_resolution=None):
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
        self._matvec_methods += ['cython', 'self']

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SB)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = {-3: (k[3:-2]-2)*(k[3:-2]+1)/k[3:-2]*np.pi,
             -1: -2*(k[1:-3]+1)**2/(k[1:-3]+2)*np.pi,
              1: (k[:-5]+1)*np.pi}
        return d

    def matvec(self, v, c, format=None, axis=0):
        format = 'cython' if format is None else format
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

        elif format == 'cython':
            cython.Matvec.CDB_matvec(v, c, axis, self[-3], self[-1], self[1])
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
    def __init__(self, test, trial, scale=1, measure=1, assemble=None, kind=None, fixed_resolution=None):
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
        self._matvec_methods += ['cython', 'self']

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        N = test[0].N
        k = np.arange(N-4, dtype=float)
        d = {-2: 2*(k[2:]-1)*(k[2:]+2)*np.pi,
              0: -4*((k+1)*(k+2)**2)/(k+3)*np.pi,
              2: 2*(k[:-2]+1)*(k[:-2]+2)*np.pi}
        return d

    def matvec(self, v, c, format=None, axis=0):
        format = 'cython' if format is None else format
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

        elif format == 'cython':
            cython.Matvec.Tridiagonal_matvec(v, c, axis, self[-2], self[0], self[2])
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
    def __init__(self, test, trial, scale=1, measure=1, assemble=None, kind=None, fixed_resolution=None):
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
        self._matvec_methods += ['cython']

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
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
        return d

    def get_solver(self):
        return ADDSolver

    def matvec(self, v, c, format=None, axis=0):
        format = 'cython' if format is None else format
        c.fill(0)
        if format == 'cython':
            cython.Matvec.ADD_matvec(v, c, axis, self[0]/self._keyscale)
            self.scale_array(c, self.scale*self._keyscale)
        else:
            format = None if format in self._matvec_methods else format
            c = super(ASDSDmat, self).matvec(v, c, format=format, axis=axis)
        return c


class ASNSNmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\phi''_j, \phi_k)_w

    where :math:`\phi_k \in` :class:`.chebyshev.bases.ShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1, assemble=None, kind=None, fixed_resolution=None):
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
        self._matvec_methods += ['numba']

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
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
        return d

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
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SD)
        N = test[0].N-4
        M = trial[0].N-2
        Q = min(M, N)
        k = np.arange(Q, dtype=float)
        d = {0: -2*(k+1)*(k+2)*np.pi,
             2: 2*(k[:min(Q, M-2)]+1)*(k[:min(Q, M-2)]+2)*np.pi}
        return d


class ATTmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (T''_j, T_k)_w

    where :math:`T_k \in` :class:`.chebyshev.bases.Orthogonal`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def __init__(self, test, trial, scale=1, measure=1, assemble=None, kind=None, fixed_resolution=None):
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
        self._matvec_methods += ['cython']

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], T)
        assert isinstance(trial[0], T)
        M, N = test[0].N, trial[0].N
        k = np.arange(min(M, N), dtype=float)
        self._keyscale = 1
        def _getkey(j):
            return self._keyscale*k[j:]*(k[j:]**2-k[:-j]**2)*np.pi/2
        d = dict.fromkeys(np.arange(2, N, 2), _getkey)
        return d

    def matvec(self, v, c, format=None, axis=0):
        format = 'cython' if format is None else format
        c.fill(0)
        if format == 'cython':
            cython.Matvec.ATT_matvec(v, c, axis)
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
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
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
        return d

class ATSNmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\psi''_j, T_k)_w

    where the test function :math:`T_k \in` :class:`.chebyshev.bases.Orthogonal`,
    the trial function :math:`\psi_j \in` :class:`.chebyshev.bases.ShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
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
        return d

class ACNCNmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\phi''_j, \phi_k)_w

    where :math:`\phi_k \in` :class:`.chebyshev.bases.CombinedShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
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
        return d

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
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], HH)
        N = test[0].N
        k = np.arange(N)
        h = get_norm_sq(test[0], trial[0], method)

        if trial[0].is_scaled():
            d = {0: -h[:-2],
                 2: np.pi/2*(k[:-4]*k[1:-3])/((k[:-4]+3)*(k[:-4]+4))}
        else:
            d = {0: -h[:-2]*k[2:]*k[1:-1],
                 2: np.pi/2*k[:-4]*k[1:-3]}
        return d

    def get_solver(self):
        return TwoDMA

class AHHHHmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\phi''_j, \phi_k)_w

    where :math:`\phi_k \in` :class:`.chebyshev.bases.Heinrichs`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], HH)
        assert isinstance(trial[0], HH)
        assert test[0].is_scaled() == trial[0].is_scaled()
        N = test[0].N
        k = np.arange(N-2, dtype=float)
        h = get_norm_sq(test[0], trial[0], method)
        ck = h*2/np.pi
        dk = ck.copy()
        dk[:2] = 0
        d = {0: -np.pi/8*ck[:-2]**2*((k+1)*(k+2)+dk[:-2]*(k-2)*(k-1)),
             2: np.pi/8*ck[:-4]*k[:-2]*k[1:-1],
            -2: np.pi/8*ck[:-4]*k[2:]*k[1:-1]}
        if test[0].is_scaled() and trial[0].is_scaled():
            d[0] *= 1/((k+2)**2*(k+1)**2)
            d[2] *= 1/((k[:-2]+1)*(k[:-2]+2)*(k[:-2]+3)*(k[:-2]+4))
            d[-2]*= 1/(k[1:-1]*k[2:]*(k[2:]+1)*(k[2:]+2))
        return d


class ASDMNmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\psi''_j, \phi_k)_w

    where the test function :math:`\phi_k \in` :class:`.chebyshev.bases.ShenDirichlet`,
    the trial function :math:`\psi_j \in` :class:`.chebyshev.bases.MikNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], MN)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = {0: -2*np.pi*np.ones(N-2),
             2: 2*np.pi*k[1:-3]/k[3:-1]}
        d[0][0] = 0
        return d

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
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
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
        return d

    def get_solver(self):
        return TwoDMA


class SSBSBmat(SpectralMatrix):
    r"""Biharmonic matrix :math:`S=(s_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        s_{kj}=(\phi''''_j, \phi_k)_w,

    where :math:`\phi_k \in` :class:`.chebyshev.bases.ShenBiharmonic`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1, assemble=None, kind=None, fixed_resolution=None):
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
        self._matvec_methods += ['cython']

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
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
        return d

    def matvec(self, v, c, format=None, axis=0):
        format = 'cython' if format is None else format
        c.fill(0)
        if format == 'cython':
            cython.Matvec.SBB_matvec(v, c, axis, self[0]/self._keyscale)
            self.scale_array(c, self.scale*self._keyscale)
        else:
            format = None if format in self._matvec_methods else format
            c = super(SSBSBmat, self).matvec(v, c, format=format, axis=axis)

        return c

# Define dictionary to hold all predefined matrices
# When looked up, missing matrices will be generated automatically
mat = SpectralMatDict({
    ((T,  0), (T , 0)): BTTmat,
    ((T,  0), (L , 0)): BTLmat,
    ((SD, 0), (SD, 0)): BSDSDmat,
    ((SB, 0), (SB, 0)): BSBSBmat,
    ((SN, 0), (SN, 0)): BSNSNmat,
    ((SD, 0), (SN, 0)): BSDSNmat,
    ((SN, 0), (SD, 0)): BSNSDmat,
    ((LD, 0), (LD, 0)): BLDLDmat,
    ((T,  0), (SN, 0)): BTSNmat,
    ((SB, 0), (SD, 0)): BSBSDmat,
    ((SB, 0), (T,  0)): BSBTmat,
    ((T,  0), (SD, 0)): BTSDmat,
    ((SD, 0), (T,  0)): BSDTmat,
    ((SD, 0), (SD, 2)): ASDSDmat,
    ((T,  0), (T , 2)): ATTmat,
    ((T,  0), (SD, 2)): ATSDmat,
    ((T,  0), (SN, 2)): ATSNmat,
    ((SN, 0), (SN, 2)): ASNSNmat,
    ((CN, 0), (CN, 2)): ACNCNmat,
    ((SB, 0), (SB, 2)): ASBSBmat,
    ((SB, 0), (SD, 2)): ASBSDmat,
    ((HH, 0), (HH, 2)): AHHHHmat,
    ((SD, 0), (MN, 2)): ASDMNmat,
    ((SN, 0), (CN, 2)): ASNCNmat,
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
    })

#mat = SpectralMatDict({})
