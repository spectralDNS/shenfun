r"""
This module contains specific inner product matrices for the different bases in
the Legendre family.

A naming convention is used for the first three capital letters for all matrices.
The first letter refers to type of matrix.

    - Mass matrices start with `B`
    - One derivative start with `C`
    - Stiffness - One derivative for test and trial - start with `A`
    - Biharmonic - Two derivatives for test and trial - start with `S`

The next two letters refer to the test and trialfunctions, respectively

    - Dirichlet:   `D`
    - Neumann:     `N`
    - Legendre:    `L`
    - Biharmonic:  `B`

As such, there are 4 mass matrices, BDDmat, BNNmat, BLLmat and BBBmat,
corresponding to the four bases above.

A matrix may consist of different types of test and trialfunctions as long as
they are all in the Legendre family. A mass matrix using Dirichlet test and
Neumann trial is named BDNmat.

All matrices in this module may be looked up using the 'mat' dictionary,
which takes test and trialfunctions along with the number of derivatives
to be applied to each. As such the mass matrix BDDmat may be looked up
as

>>> import numpy as np
>>> from shenfun.legendre.matrices import mat
>>> from shenfun.legendre.bases import ShenDirichlet as SD
>>> B = mat[((SD, 0), (SD, 0))]

and an instance of the matrix can be created as

>>> B0 = SD(10)
>>> BM = B((B0, 0), (B0, 0))
>>> d = {-2: np.array([-0.4, -0.28571429, -0.22222222, -0.18181818, -0.15384615, -0.13333333]),
...       0: np.array([2.4, 0.95238095, 0.62222222, 0.46753247, 0.37606838, 0.31515152, 0.27149321, 0.23859649]),
...       2: np.array([-0.4, -0.28571429, -0.22222222, -0.18181818, -0.15384615, -0.13333333])}
>>> [np.all(abs(BM[k]-v) < 1e-7) for k, v in d.items()]
[True, True, True]

However, this way of creating matrices is not reccommended use. It is far
more elegant to use the TrialFunction/TestFunction interface, and to
generate the matrix as an inner product:

>>> from shenfun import TrialFunction, TestFunction, inner
>>> u = TrialFunction(B0)
>>> v = TestFunction(B0)
>>> BM = inner(u, v)
>>> [np.all(abs(BM[k]-v) < 1e-7) for k, v in d.items()]
[True, True, True]

To see that this is in fact the BDDmat:

>>> print(BM.__class__)
<class 'shenfun.legendre.matrices.BDDmat'>

"""
from __future__ import division

#__all__ = ['mat']

import functools
import numpy as np
import sympy as sp
from shenfun.matrixbase import SpectralMatrix
from shenfun.la import TDMA as neumann_TDMA
from shenfun.optimization import cython
from .la import TDMA
from . import bases

# Short names for instances of bases
LB = bases.Orthogonal
SD = bases.ShenDirichlet
SB = bases.ShenBiharmonic
SN = bases.ShenNeumann
SU = bases.UpperDirichlet
CD = bases.BCDirichlet
CB = bases.BCBiharmonic

x = sp.symbols('x', real=True)
xp = sp.symbols('x', real=True, positive=True)

#pylint: disable=unused-variable, redefined-builtin, bad-continuation


class BLLmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (L_j, L_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N

    and :math:`L_k` is the Legendre basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], LB)
        assert isinstance(trial[0], LB)
        N = test[0].N
        k = np.arange(N, dtype=np.float)
        d = {0: 2./(2.*k+1)}
        if test[0].quad == 'GL':
            d[0][-1] = 2./(N-1)
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)

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


class BDDmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\psi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N-2, dtype=np.float)
        d = {-2: -2./(2*k[2:] + 1),
              0: 2./(2.*k+1) + 2./(2*k+5)}

        if test[0].quad == 'GL':
            d[0][-1] = 2./(2*(N-3)+1) + 2./(N-1)

        if test[0].is_scaled():
            d[0] /= (4*k+6)
            d[-2] /= (np.sqrt(4*k[2:]+6)*np.sqrt(4*k[:-2]+6))

        d[2] = d[-2]
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
        self.solve = TDMA(self)


class BNNmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\psi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Neumann basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SN)
        N = test[0].N
        k = np.arange(N-2, dtype=np.float)
        alpha = k*(k+1)/(k+2)/(k+3)
        d0 = 2./(2*k+1)
        d = {0: d0 + alpha**2*2./(2*(k+2)+1),
             2: -d0[2:]*alpha[:-2]}
        if test[0].quad == 'GL':
            d[0][-1] = d0[-1] + alpha[-1]**2*2./(N-1)
        d[-2] = d[2]
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
        #self.solve = neumann_TDMA(self)


class BBBmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\psi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Biharmonic basis function.

    """
    def __init__(self, test, trial, measure=1):
        from shenfun.la import PDMA
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        N = test[0].N
        k = np.arange(N, dtype=np.float)
        gk = (2*k+3)/(2*k+7)
        hk = -(1+gk)
        ek = 2./(2*k+1)
        if test[0].quad == 'GL':
            ek[-1] = 2./(N-1)
        d = {0: ek[:-4] + hk[:-4]**2*ek[2:-2] + gk[:-4]**2*ek[4:],
             2: hk[:-6]*ek[2:-4] + gk[:-6]*hk[2:-4]*ek[4:-2],
             4: gk[:-8]*ek[4:-4]}
        d[-2] = d[2]
        d[-4] = d[4]
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)
        self.solve = PDMA(self)


class BDLmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (L_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], LB)
        N = test[0].N
        k = np.arange(N, dtype=np.float)
        sc = np.ones(N)
        if test[0].is_scaled():
            sc = 1. / np.sqrt(4*k+6)
        d = {2: -2./(2*k[2:] + 1)*sc[:-2],
             0: 2./(2.*k[:-2]+1)*sc[:-2]}
        if test[0].quad == 'GL':
            d[2][-1] = -2./(N-1)*sc[N-3]
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)


class BLDmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\psi_j, L_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N

    and :math:`\psi_j` is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], LB)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=np.float)
        sc = np.ones(N)
        if trial[0].is_scaled():
            sc = 1. / np.sqrt(4*k+6)

        d = {-2: -2./(2*k[2:] + 1)*sc[:-2],
             0: 2./(2.*k[:-2]+1)*sc[:-2]}
        if test[0].quad == 'GL':
            d[-2][-1] = -2./(N-1)*sc[-3]
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)


class ADDmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\psi'_j, \psi'_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N-2, dtype=np.float)
        if not test[0].is_scaled():
            d = {0: 4*k+6}
        else:
            d = {0: 1}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

    def solve(self, b, u=None, axis=0):
        N = self.shape[0] + 2
        assert N == b.shape[axis]
        s = self.trialfunction[0].slice()

        if u is None:
            u = b
        else:
            assert u.shape == b.shape

        if not self.trialfunction[0].is_scaled():

            # Move axis to first
            if axis > 0:
                u = np.moveaxis(u, axis, 0)
                if u is not b:
                    b = np.moveaxis(b, axis, 0)

            bs = b[s]
            us = u[s]
            d = 1./self[0]
            sl = [np.newaxis]*bs.ndim
            sl[0] = slice(None)
            us[:] = bs*d[tuple(sl)]
            u /= self.scale
            self.testfunction[0].bc.set_boundary_dofs(u, True)

            if axis > 0:
                u = np.moveaxis(u, 0, axis)
                if u is not b:
                    b = np.moveaxis(b, axis, 0)
        else:
            ss = [slice(None)]*b.ndim
            ss[axis] = s
            ss = tuple(ss)
            u[ss] = b[ss]
            u /= self.scale
            self.testfunction[0].bc.set_boundary_dofs(u, True)

        return u


class ANNmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\psi'_j, \psi'_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Neumann basis function.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SN)
        N = test[0].N
        k = np.arange(N-2, dtype=np.float)
        alpha = k*(k+1)/(k+2)/(k+3)
        d0 = 2./(2*k+1)
        d = {0: d0*alpha*(k+0.5)*((k+2)*(k+3)-k*(k+1))}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

    def solve(self, b, u=None, axis=0):
        N = self.shape[0] + 2
        assert N == b.shape[axis]
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
        d = np.ones(self.shape[0])
        d[1:] = 1./self[0][1:]
        sl = [np.newaxis]*bs.ndim
        sl[0] = slice(None)
        us[:] = bs*d[tuple(sl)]
        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if u is not b:
                b = np.moveaxis(b, axis, 0)

        u /= self.scale
        return u


class ABBmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\psi'_j, \psi'_k)_w

    where

    .. math::

        j = 0, 1, ..., N-4 \text{ and } k = 0, 1, ..., N-4

    and :math:`\psi_k` is the Shen Legendre Biharmonic basis function.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        N = test[0].N
        k = np.arange(N-4, dtype=np.float)
        gk = (2*k+3)/(2*k+7)
        d = {0: 2*(2*k+3)*(1+gk),
             2: -2*(2*k[:-2]+3)}
        d[-2] = d[2]
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

class GLLmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        B_{kj} = (L_j'', L_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N

    and :math:`L_k` is the Legendre basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], LB)
        assert isinstance(trial[0], LB)
        N = test[0].N
        k = np.arange(N, dtype=np.float)
        d = {}
        for j in range(2, N, 2):
            jj = j if trial[1] else -j
            d[jj] = (k[:-j]+0.5)*(k[j:]*(k[j:]+1) - k[:-j]*(k[:-j]+1))*2./(2*k[:-j]+1)
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)

    def matvec(self, v, c, format='cython', axis=0):
        c.fill(0)
        trial = self.trialfunction[1]
        if format == 'cython' and v.ndim == 3 and trial:
            cython.Matvec.GLL_matvec3D_ptr(v, c, axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 2 and trial:
            cython.Matvec.GLL_matvec2D_ptr(v, c, axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 1 and trial:
            cython.Matvec.GLL_matvec(v, c)
            self.scale_array(c)
        else:
            c = super(GLLmat, self).matvec(v, c, format=format, axis=axis)
        return c

class SBBmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\psi''_j, \psi''_k)_w

    where

    .. math::

        j = 0, 1, ..., N-4 \text{ and } k = 0, 1, ..., N-4

    and :math:`\psi_k` is the Shen Legendre Biharmonic basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        N = test[0].N
        k = np.arange(N-4, dtype=np.float)
        d = {0: 2*(2*k+3)**2*(2*k+5)}
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)


class CLLmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (\psi'_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N

    and :math:`\psi_k` is the orthogonal Legendre basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], LB)
        assert isinstance(trial[0], LB)
        N = test[0].N
        d = {}
        for i in range(1, N, 2):
            d[i] = 2
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)

    def matvec(self, v, c, format='self', axis=0):
        c.fill(0)
        if format == 'self':
            if axis > 0:
                c = np.moveaxis(c, axis, 0)
                v = np.moveaxis(v, axis, 0)
            ve = v[-2:0:-2].cumsum(axis=0)
            vo = v[-1:0:-2].cumsum(axis=0)
            c[-3::-2] = ve*2
            c[-2::-2] = vo*2
            if axis > 0:
                c = np.moveaxis(c, 0, axis)
                v = np.moveaxis(v, 0, axis)
            self.scale_array(c)

        elif format == 'cython' and v.ndim == 3:
            cython.Matvec.CLL_matvec3D_ptr(v, c, axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 2:
            cython.Matvec.CLL_matvec2D_ptr(v, c, axis)
            self.scale_array(c)
        elif format == 'cython' and v.ndim == 1:
            cython.Matvec.CLL_matvec(v, c)
            self.scale_array(c)
        else:
            c = super(CLLmat, self).matvec(v, c, format=format, axis=axis)
        return c

class CLLmatT(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (\psi'_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N

    and :math:`\psi_k` is the orthogonal Legendre basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], LB)
        assert isinstance(trial[0], LB)
        N = test[0].N
        d = {}
        for i in range(-1, -N, -2):
            d[i] = 2
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)


class CLDmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (\psi'_j, L_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N

    and :math:`\psi_k` is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], LB)
        assert isinstance(trial[0], SD)
        N = test[0].N
        d = {-1: -2}
        if trial[0].is_scaled():
            k = np.arange(N-2, dtype=np.float)
            d[-1] = -2. / np.sqrt(4*k+6)
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)


class CDLmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (L_j, \psi'_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], LB)
        N = test[0].N
        d = {1: -2}
        if test[0].is_scaled():
            k = np.arange(N-2, dtype=np.float)
            d[1] = -2. / np.sqrt(4*k+6)
        SpectralMatrix.__init__(self, d, test, trial, measure=measure)


class CDDmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (\psi'_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        N = test[0].N
        d = {-1: -2, 1: 2}
        if trial[0].is_scaled():
            k = np.arange(N-2, dtype=np.float)
            d[-1] = -2. / np.sqrt(4*k[:-1]+6)
            d[1] = 2. / np.sqrt(4*k[:-1]+6)
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class ADDrp1mat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        A_{kj} = \int_{-1}^{1} \psi_j'(x) \psi_k'(x) (1+x) dx

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        assert test[0].quad == 'LG'

        k = np.arange(test[0].N-2)
        d = {0: 4*k+6, 1: 2*k[:-1]+4, -1: 2*k[:-1]+4}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class ADD2rp1mat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        A_{kj} = \int_{-1}^{1} \psi_j(x) \psi_k''(x) (1+x) dx

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        assert test[0].quad == 'LG'

        k = np.arange(test[0].N-2)
        d = {0: -(4*k+6), 1: -(2*k[:-1]+6), -1: -(2*k[:-1]+2)}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class ADD2Trp1mat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        A_{kj} = \int_{-1}^{1} \psi_j''(x) \psi_k(x) (1+x) dx

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        assert test[0].quad == 'LG'

        k = np.arange(test[0].N-2)
        d = {0: -(4*k+6), -1: -(2*k[:-1]+6), 1: -(2*k[:-1]+2)}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class AUUrp1mat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        A_{kj} = \int_{-1}^{1} \psi_j'(x) \psi_k'(x) (1+x) dx

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Upper Dirichlet basis function.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SU)
        assert isinstance(trial[0], SU)
        assert test[0].quad == 'LG'
        k = np.arange(test[0].N-1)
        d = {0: 2*k+2}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class BUUrp1mat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        B_{kj} = \int_{-1}^{1} \psi_j(x) \psi_k(x) (1+x) dx

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Upper Dirichlet basis function.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SU)
        assert isinstance(trial[0], SU)
        assert test[0].quad == 'LG'
        k = np.arange(test[0].N-1)
        d = {0: 2*k+2}
        d = {0: 4*(k+1)/(2*k+1)/(2*k+3),
             1: 4/(2*k[:-1]+1)/(2*k[:-1]+3)/(2*k[:-1]+5),
             2: -2*(k[:-2]+2)/(2*k[:-2]+3)/(2*k[:-2]+5)}
        d[-1] = d[1]
        d[-2] = d[2]
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class BDD1orp1mat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        A_{kj} = \int_{-1}^{1} \psi_j(x) \psi_k(x) 1/(1+x) dx

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        assert test[0].quad == 'LG'

        k = np.arange(test[0].N-2)
        d = {0: 2*(2*k+3)/(k+1)/(k+2), 1: -2/(k[:-1]+2), -1: -2/(k[:-1]+2)}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class BDDrp1mat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        A_{kj} = \int_{-1}^{1} \psi_j(x) \psi_k(x) (1+x) dx

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        assert test[0].quad == 'LG'

        k = np.arange(test[0].N-2)
        d = {0: 2/(2*k+1)+2/(2*k+5),
             1: 2/(2*k[:-1]+1)/(2*k[:-1]+5) + 2*(k[:-1]+3)/(2*k[:-1]+5)/(2*k[:-1]+7),
             2: -2/(2*k[:-2]+5),
             3: -2*(k[:-3]+3)/(2*k[:-3]+5)/(2*k[:-3]+7)}
        d[-1] = d[1]
        d[-2] = d[2]
        d[-3] = d[3]
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class BCDmat(SpectralMatrix):
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
        assert isinstance(trial[0], CD)
        N = test[0].N
        k = np.arange(N-2, dtype=np.float)
        if not test[0].is_scaled():
            d = {0: np.array([1, 1./3.]),
                 1: np.array([1.0]),
                 -1: np.array([-1./3., 0])}
        else:
            d = {0: np.array([1./np.sqrt(6.), 1./3./np.sqrt(10.)]),
                 1: np.array([1./np.sqrt(6.)]),
                 -1: np.array([-1./3./np.sqrt(10.), 0])}

        SpectralMatrix.__init__(self, d, test, trial, measure=measure)


class BCBmat(SpectralMatrix):
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
        assert isinstance(trial[0], CB)
        N = test[0].N
        k = np.arange(N-4, dtype=np.float)
        d = {0: np.array([1, 4/9, -1/15, 1/35]),
             1: np.array([1, -1/9, 1/15]),
             2: np.array([3/7, -1/9]),
             3: np.array([-3/7]),
            -1: np.array([-4/9, 0, 1/35, 0]),
            -2: np.array([0, -1/35, 0, 0]),
            -3: np.array([1/35, 0, 0, 0])}

        SpectralMatrix.__init__(self, d, test, trial, measure=measure)


class _Legmatrix(SpectralMatrix):
    def __init__(self, test, trial, measure=1):
        SpectralMatrix.__init__(self, {}, test, trial, measure=measure)


class _LegMatDict(dict):
    """Dictionary of inner product matrices

    Matrices that are missing keys are generated from Vandermonde type
    computations.

    """

    def __missing__(self, key):
        measure = 1 if len(key) == 2 else key[3]
        c = functools.partial(_Legmatrix, measure=measure)
        self[key] = c
        return c

    def __getitem__(self, key):
        matrix = dict.__getitem__(self, key)
        return matrix

mat = _LegMatDict({
    ((LB, 0), (LB, 0)): BLLmat,
    ((LB, 0), (LB, 1)): CLLmat,
    ((LB, 1), (LB, 0)): CLLmatT,
    ((LB, 0), (SD, 1)): CLDmat,
    ((SD, 1), (LB, 0)): CDLmat,
    ((SD, 0), (SD, 1)): CDDmat,
    ((SD, 1), (SD, 0)): functools.partial(CDDmat, scale=-1.),
    ((SD, 0), (SD, 0)): BDDmat,
    ((SB, 0), (SB, 0)): BBBmat,
    ((SN, 0), (SN, 0)): BNNmat,
    ((SD, 0), (LB, 0)): BDLmat,
    ((LB, 0), (SD, 0)): BLDmat,
    ((SD, 1), (SD, 1)): ADDmat,
    ((SD, 2), (SD, 0)): functools.partial(ADDmat, scale=-1.),
    ((SD, 0), (SD, 2)): functools.partial(ADDmat, scale=-1.),
    ((SN, 1), (SN, 1)): ANNmat,
    ((SN, 2), (SN, 0)): functools.partial(ANNmat, scale=-1.),
    ((SN, 0), (SN, 2)): functools.partial(ANNmat, scale=-1.),
    ((LB, 2), (LB, 0)): GLLmat,
    ((LB, 0), (LB, 2)): GLLmat,
    ((SB, 2), (SB, 2)): SBBmat,
    ((SB, 1), (SB, 1)): ABBmat,
    ((SB, 0), (SB, 2)): functools.partial(ABBmat, scale=-1.),
    ((SB, 2), (SB, 0)): functools.partial(ABBmat, scale=-1.),
    ((SB, 0), (SB, 4)): SBBmat,
    ((SB, 4), (SB, 0)): SBBmat,
    ((SD, 1), (SD, 1), (-1, 1), 1+x): ADDrp1mat,
    ((SD, 0), (SD, 2), (-1, 1), 1+x): ADD2rp1mat,
    ((SD, 2), (SD, 0), (-1, 1), 1+x): ADD2Trp1mat,
    ((SD, 0), (SD, 2), (0, 1), xp): functools.partial(ADD2rp1mat, scale=0.5),
    ((SD, 2), (SD, 0), (0, 1), xp): functools.partial(ADD2Trp1mat, scale=0.5),
    ((SD, 1), (SD, 1), (0, 1), xp): functools.partial(ADDrp1mat, scale=0.5),
    ((SD, 0), (SD, 0), (-1, 1), 1+x): BDDrp1mat,
    ((SD, 0), (SD, 0), (0, 1), xp): functools.partial(BDDrp1mat, scale=0.5),
    ((SD, 0), (SD, 0), (-1, 1), 1/(1+x)): BDD1orp1mat,
    ((SD, 0), (SD, 0), (0, 1), 1/xp): functools.partial(BDD1orp1mat, scale=2),
    ((SU, 1), (SU, 1), (-1, 1), 1+x): AUUrp1mat,
    ((SU, 1), (SU, 1), (0, 1), xp): functools.partial(AUUrp1mat, scale=0.5),
    ((SU, 0), (SU, 0), (-1, 1), 1+x): BUUrp1mat,
    ((SU, 0), (SU, 0), (0, 1), xp): functools.partial(BUUrp1mat, scale=0.5),
    ((SD, 0), (CD, 0)): BCDmat,
    ((SB, 0), (CB, 0)): BCBmat,
    })
