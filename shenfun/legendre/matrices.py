r"""
This module contains specific inner product matrices for the different bases in
the Legendre family.

A naming convention is used for the first three capital letters for all matrices.
The first letter refers to type of matrix.

    - Mass matrices start with `B`
    - One derivative start with `C`
    - Stiffness - One derivative for test and trial - start with `A`
    - Biharmonic - Two derivatives for test and trial - start with `S`

A matrix may consist of different types of test and trialfunctions. The next
letters in the matrix name uses the 'short_name' method for all these different
bases, see legendre.bases.py.

So a mass matrix using ShenDirichlet test and ShenNeumann trial is named
BSDSNmat.

All matrices in this module may be looked up using the 'mat' dictionary,
which takes test and trialfunctions along with the number of derivatives
to be applied to each. As such the mass matrix BSDSDmat may be looked up
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

To see that this is in fact the BSDSDmat:

>>> print(BM.__class__)
<class 'shenfun.legendre.matrices.BSDSDmat'>

"""
from __future__ import division

#__all__ = ['mat']

import functools
import numpy as np
import sympy as sp
from shenfun.matrixbase import SpectralMatrix, SpectralMatDict
from shenfun.spectralbase import get_norm_sq
from shenfun.optimization import cython
from shenfun.la import TDMA, PDMA
from . import bases

# Short names for instances of bases
L  = bases.Orthogonal
SD = bases.ShenDirichlet
SB = bases.ShenBiharmonic
SN = bases.ShenNeumann
UD = bases.UpperDirichlet
LD = bases.LowerDirichlet
DN = bases.DirichletNeumann
ND = bases.NeumannDirichlet
BF = bases.BeamFixedFree
CB = bases.CompositeBase
P1 = bases.Phi1
P2 = bases.Phi2
P3 = bases.Phi3
P4 = bases.Phi4

BCG = bases.BCGeneric

x = sp.symbols('x', real=True)
xp = sp.symbols('x', real=True, positive=True)

#pylint: disable=unused-variable, redefined-builtin, bad-continuation

class BLLmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(L_j, L_k),

    where :math:`L_k \in` :class:`.legendre.bases.Orthogonal`, and test
    and trial spaces have dimensions of M and N, respectively.
    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], L)
        assert isinstance(trial[0], L)
        return {0: get_norm_sq(test[0], trial[0], method)}


class BSDSDmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\phi_j, \phi_k),

    where :math:`\phi_k \in` :class:`.legendre.bases.ShenDirichlet`, and test
    and trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        d0 = get_norm_sq(test[0], trial[0], method)
        d = {0: d0[:-2]+d0[2:], -2: -d0[2:-2]}

        if test[0].is_scaled():
            k = np.arange(test[0].N-2)
            d[0] /= (4*k+6)
            d[-2] /= (np.sqrt(4*k[2:]+6)*np.sqrt(4*k[:-2]+6))

        d[2] = d[-2].copy()
        return d

    def get_solver(self):
        return TDMA

class BSNSNmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\phi_j, \phi_k),

    where :math:`\phi_k \in` :class:`.legendre.bases.ShenNeumann`, and test
    and trial spaces have dimensions of M and N, respectively.
    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SN)
        N = test[0].N
        k = np.arange(N-2, dtype=float)
        alpha = k*(k+1)/(k+2)/(k+3)
        d0 = get_norm_sq(test[0], trial[0], method)
        d = {
            0: d0[:-2] + alpha**2*d0[2:],
            2: -alpha[:-2]*d0[2:-2]
        }
        d[-2] = d[2].copy()
        return d


class BSBSBmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\phi_j, \phi_k),

    where :math:`\phi_k \in` :class:`.legendre.bases.ShenBiharmonic`, and test
    and trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        d0 = get_norm_sq(test[0], trial[0], method)
        i = np.arange(test[0].dim())
        d = {
            0: (-4*i - 10)**2*d0[2:-2]/(2*i + 7)**2 + (2*i + 3)**2*d0[4:]/(2*i + 7)**2 + d0[:-4],
            2: (-4*i[:-2] - 18)*(2*i[:-2] + 3)*d0[4:-2]/((2*i[:-2] + 7)*(2*i[:-2] + 11)) + (-4*i[:-2] - 10)*d0[2:-4]/(2*i[:-2] + 7),
            4: (2*i[:-4] + 3)*d0[4:-4]/(2*i[:-4] + 7)
        }
        d[-2] = d[2].copy()
        d[-4] = d[4].copy()
        return d

    def get_solver(self):
        return PDMA


class BSDLmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(L_j, \phi_k),

    where the test function :math:`\phi_k \in` :class:`.legendre.bases.ShenDirichlet`, the
    trial function :math:`L_j \in` :class:`.legendre.bases.Orthogonal`, and test
    and trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], L)
        d0 = get_norm_sq(test[0], trial[0], method)
        S = test[0].stencil_matrix()
        d = {
            0: d0[:-2]*S[0][:-2] if isinstance(S[0], np.ndarray) else d0[:-2]*S[0],
            2: d0[2:]*S[2]
        }
        return d


class BLSDmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\psi_j, L_k),

    where the test function :math:`L_k \in` :class:`.legendre.bases.Orthogonal`, the
    trial function :math:`\psi_j \in` :class:`.legendre.bases.ShenDirichlet`, and test
    and trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], L)
        assert isinstance(trial[0], SD)
        d0 = get_norm_sq(test[0], trial[0], method)
        S = trial[0].stencil_matrix()
        d = {
            0: d0[:-2]*S[0][:-2] if isinstance(S[0], np.ndarray) else d0[:-2]*S[0],
            -2: d0[2:]*S[2]
        }
        return d

class BDNDNmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\phi_j, \phi_k),

    where :math:`\phi_k \in` :class:`.legendre.bases.DirichletNeumann`, and test
    and trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], DN)
        assert isinstance(trial[0], DN)
        d0 = get_norm_sq(test[0], trial[0], method)
        i = np.arange(test[0].N-2)
        d = {
            0: (2*i + 3)**2*d0[1:-1]/(i**2 + 4*i + 4)**2 + (-i**2 - 2*i - 1)**2*d0[2:]/(i**2 + 4*i + 4)**2 + d0[:-2],
            1: (2*i[:-1] + 3)*d0[1:-2]/(i[:-1]**2 + 4*i[:-1] + 4) + (2*i[:-1] + 5)*(-i[:-1]**2 - 2*i[:-1] - 1)*d0[2:-1]/((4*i[:-1] + (i[:-1] + 1)**2 + 8)*(i[:-1]**2 + 4*i[:-1] + 4)),
            2: (-i[:-2]**2 - 2*i[:-2] - 1)*d0[2:-2]/(i[:-2]**2 + 4*i[:-2] + 4)
        }
        d[-1] = d[1].copy()
        d[-2] = d[2].copy()
        return d


class ASDSDmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\phi'_j, \phi'_k)

    where :math:`\phi_k \in` :class:`.legendre.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N-2, dtype=float)
        if not test[0].is_scaled():
            d = {0: 4*k+6}
        else:
            d = {0: 1}
        return d


class ASNSNmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\phi'_j, \phi'_k)

    where :math:`\phi_k \in` :class:`.legendre.bases.ShenNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SN)
        N = test[0].N
        k = np.arange(N-2, dtype=float)
        alpha = k*(k+1)/(k+2)/(k+3)
        d0 = 2./(2*k+1)
        d = {0: d0*alpha*(k+0.5)*((k+2)*(k+3)-k*(k+1))}
        return d


class ASBSBmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\phi'_j, \phi'_k)

    where :math:`\phi_k \in` :class:`.legendre.bases.ShenBiharmonic`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        N = test[0].N
        k = np.arange(N-4, dtype=float)
        gk = (2*k+3)/(2*k+7)
        d = {0: 2*(2*k+3)*(1+gk),
             2: -2*(2*k[:-2]+3)}
        d[-2] = d[2].copy()
        return d

class ADNDNmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\phi'_j, \phi'_k)

    where :math:`\phi_k \in` :class:`.legendre.bases.DirichletNeumann`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], DN)
        assert isinstance(trial[0], DN)
        N = test[0].N
        k = np.arange(N-2, dtype=float)
        d = {0: ((k+1)/(k+2))**2*((k+2)*(k+3)- k*(k+1))}
        return d


class SBFBFmat(SpectralMatrix):
    r"""Biharmonic matrix :math:`S=(s_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        s_{kj} = (\phi''_j, \phi''_k)

    where :math:`\phi_k \in` :class:`.legendre.bases.BeamFixedFree`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], BF)
        assert isinstance(trial[0], BF)
        N = test[0].N
        k = np.arange(N-4, dtype=float)
        f4 = (((k+1)/(k+3))*((k+2)/(k+4)))**2*(2*k+3)/(2*k+7)
        d = {0: f4*(k+2.5)*((k+4)*(k+5)-(k+2)*(k+3))*((k+2)*(k+3)-k*(k+1))}
        return d

class ALLmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} &= (L''_j, L_k), \text{ or } \\
        a_{kj} &= (L_j, L''_k)

    where :math:`L_k \in` :class:`.legendre.bases.Orthogonal`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1, assemble=None, kind=None, fixed_resolution=None):
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
        #self._matvec_methods += ['cython']

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], L)
        assert isinstance(trial[0], L)
        M, N = test[0].N, trial[0].N
        k = np.arange(min(M, N), dtype=float)
        self._keyscale = 1
        def _getkey(i):
            j = abs(i)
            return self._keyscale*((k[:-j]+0.5)*(k[j:]*(k[j:]+1) - k[:-j]*(k[:-j]+1))*2./(2*k[:-j]+1))

        if trial[1]:
            d = dict.fromkeys(np.arange(2, N, 2), _getkey)
        else:
            d = dict.fromkeys(-np.arange(2, N, 2), _getkey)
        return d

    #def matvec(self, v, c, format='cython', axis=0):
    #    c.fill(0)
    #    if format == 'cython':
    #        cython.Matvec.GLL_matvec(v, c, axis)
    #        self.scale_array(c, self.scale*self._keyscale)
    #    else:
    #        format = None if format in self._matvec_methods else format
    #        c = super(ALLmat, self).matvec(v, c, format=format, axis=axis)
    #    return c

class SSBSBmat(SpectralMatrix):
    r"""Biharmonic matrix :math:`S=(s_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        s_{kj} = (\phi''_j, \phi''_k)

    where :math:`\phi_k \in` :class:`.legendre.bases.ShenBiharmonic`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        M, N = test[0].dim(), trial[0].dim()
        k = np.arange(min(M, N), dtype=float)
        d = {0: 2*(2*k+3)**2*(2*k+5)}
        return d


class CLLmat(SpectralMatrix):
    r"""Derivative matrix :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(L'_j, L_k),

    where :math:`L_k \in` :class:`.legendre.bases.Orthogonal`, and test
    and trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1, assemble=None, kind=None, fixed_resolution=None):
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)
        self._matvec_methods += ['cython', 'self']

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], L)
        assert isinstance(trial[0], L)
        N = trial[0].N
        self._keyscale = 1
        def _getkey(i):
            return 2*self._keyscale
        d = dict.fromkeys(np.arange(1, N, 2), _getkey)
        return d

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
            self.scale_array(c, self.scale*self._keyscale)

        elif format == 'cython':
            cython.Matvec.CLL_matvec(v, c, axis)
            self.scale_array(c, self.scale*self._keyscale)
        else:
            format = None if format in self._matvec_methods else format
            c = super(CLLmat, self).matvec(v, c, format=format, axis=axis)
        return c

class CLLmatT(SpectralMatrix):
    r"""Derivative matrix :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(L_j, L'_k),

    where :math:`L_k \in` :class:`.legendre.bases.Orthogonal`, and test
    and trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], L)
        assert isinstance(trial[0], L)
        N = trial[0].N
        self._keyscale = 1
        def _getkey(i):
            return 2*self._keyscale

        d = dict.fromkeys(-np.arange(1, N, 2), _getkey)
        return d

class CLSDmat(SpectralMatrix):
    r"""Derivative matrix :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(\psi'_j, L_k),

    where the test function :math:`L_k \in` :class:`.legendre.bases.Orthogonal`, the trial
    function :math:`\psi_j \in` :class:`.legendre.bases.ShenDirichlet`, and test
    and trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], L)
        assert isinstance(trial[0], SD)
        N = test[0].N
        d = {-1: -2}
        if trial[0].is_scaled():
            k = np.arange(N-2, dtype=float)
            d[-1] = -2. / np.sqrt(4*k+6)
        return d


class CSDLmat(SpectralMatrix):
    r"""Derivative matrix :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(L'_j, \phi_k),

    where the test function :math:`\phi_k \in` :class:`.legendre.bases.ShenDirichlet`, the trial
    function :math:`L_j \in` :class:`.legendre.bases.Orthogonal`, and test
    and trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], L)
        N = test[0].N
        d = {1: -2}
        if test[0].is_scaled():
            k = np.arange(N-2, dtype=float)
            d[1] = -2. / np.sqrt(4*k+6)
        return d


class CSDSDmat(SpectralMatrix):
    r"""Derivative matrix :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(\phi'_j, \phi_k),

    where :math:`\phi_k \in` :class:`.legendre.bases.ShenDirichlet`, and test
    and trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        N = test[0].N
        d = {-1: -2, 1: 2}
        if trial[0].is_scaled():
            k = np.arange(N-2, dtype=float)
            d[-1] = -2/np.sqrt(4*k[:-1]+6)
            d[1] = 2/np.sqrt(4*k[:-1]+6)
        return d

class CSDSDTmat(SpectralMatrix):
    r"""Derivative matrix :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(\phi_j, \phi'_k),

    where :math:`\phi_k \in` :class:`.legendre.bases.ShenDirichlet`, and test
    and trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        N = test[0].N
        d = {-1: 2, 1: -2}
        if trial[0].is_scaled():
            k = np.arange(N-2, dtype=float)
            d[-1] = 2/np.sqrt(4*k[:-1]+6)
            d[1] = -2/np.sqrt(4*k[:-1]+6)
        return d


class ASDSDrp1mat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\phi'_j, (1+x)\phi'_k)

    where :math:`\phi_k \in` :class:`.legendre.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        assert test[0].quad == 'LG'

        k = np.arange(test[0].N-2)
        d = {0: 4*k+6, 1: 2*k[:-1]+4, -1: 2*k[:-1]+4}
        return d


class ASDSD2rp1mat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\phi_j, (1+x)\phi''_k)

    where :math:`\phi_k \in` :class:`.legendre.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        k = np.arange(test[0].N-2)
        d = {0: -(4*k+6), 1: -(2*k[:-1]+6), -1: -(2*k[:-1]+2)}
        return d


class ASDSD2Trp1mat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\phi''_j, (1+x)\phi_k)

    where :math:`\phi_k \in` :class:`.legendre.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        k = np.arange(test[0].N-2)
        d = {0: -(4*k+6), -1: -(2*k[:-1]+6), 1: -(2*k[:-1]+2)}
        return d


class AUDUDrp1mat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\phi'_j, (1+x)\phi'_k)

    where :math:`\phi_k \in` :class:`.legendre.bases.UpperDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.
    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], UD)
        assert isinstance(trial[0], UD)
        k = np.arange(test[0].N-1)
        d = {0: 2*k+2}
        return d

class AUDUDrp1smat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\phi'_j, (1+x)^2\phi'_k)

    where :math:`\phi_k \in` :class:`.legendre.bases.UpperDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], UD)
        assert isinstance(trial[0], UD)
        k = np.arange(test[0].N-1)
        d = {0: 2*(k+1)**2*(1/(2*k+1)+1/(2*k+3)),
             1: 2*k[1:]*(k[1:]+1)/(2*k[1:]+1)}
        d[-1] = d[1].copy()
        return d

class GUDUDrp1smat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj} = (\phi_j, (1+x)^2\phi''_k)

    where :math:`\phi_k \in` :class:`.legendre.bases.UpperDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], UD)
        assert isinstance(trial[0], UD)
        k = np.arange(test[0].N-1)
        d = {0: -2*(k+1)*((k-1)/(2*k+1) + (k+3)/(2*k+3)),
             1: -2*(k[1:]+1)*(k[1:]+2)/(2*k[1:]+1),
             -1: -2*k[:-1]*(k[:-1]+1)/(2*k[:-1]+3)}
        return d

class BUDUDrp1smat(SpectralMatrix):
    r"""Matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj} = (\phi_j, (1+x)^2\phi_k)

    where :math:`\phi_k \in` :class:`.legendre.bases.UpperDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        from sympy import KroneckerDelta
        i, j = sp.symbols('i,j', real=True, integer=True)
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], UD)
        assert isinstance(trial[0], UD)
        k = np.arange(test[0].N-1)
        LL = get_norm_sq(test[0], trial[0], method)
        class h(sp.Function):
            @classmethod
            def eval(cls, x):
                if x.is_Number:
                    return LL[int(x)]

        N = test[0].dim()
        s = i*(i - 1)*(KroneckerDelta(j, i - 2) - KroneckerDelta(i - 2, j + 1))*h(i - 2)/((2*i - 1)*(2*i + 1)) - i*(i + 1)*(KroneckerDelta(j, i - 1) - KroneckerDelta(i - 1, j + 1))*h(i - 1)/((2*i + 1)*(2*i + 3)) + 2*i*(KroneckerDelta(j, i - 1) - KroneckerDelta(i - 1, j + 1))*h(i - 1)/(2*i + 1) + (i + 1)*(i + 2)*(KroneckerDelta(j, i + 2) - KroneckerDelta(i + 2, j + 1))*h(i + 2)/((2*i + 1)*(2*i + 3)) - 2*(i + 1)*(KroneckerDelta(i, j) - KroneckerDelta(i, j + 1))*h(i)/(2*i + 3) + 2*(i + 1)*(KroneckerDelta(j, i + 1) - KroneckerDelta(i + 1, j + 1))*h(i + 1)/(2*i + 1) - (i + 2)*(i + 3)*(KroneckerDelta(j, i + 3) - KroneckerDelta(i + 3, j + 1))*h(i + 3)/((2*i + 3)*(2*i + 5)) - 2*(i + 2)*(KroneckerDelta(j, i + 2) - KroneckerDelta(i + 2, j + 1))*h(i + 2)/(2*i + 3) + (KroneckerDelta(i, j) - KroneckerDelta(i, j + 1))*(i**2/((2*i - 1)*(2*i + 1)) + (i + 1)**2/((2*i + 1)*(2*i + 3)) + 1)*h(i) - (KroneckerDelta(j, i + 1) - KroneckerDelta(i + 1, j + 1))*((i + 1)**2/((2*i + 1)*(2*i + 3)) + (i + 2)**2/((2*i + 3)*(2*i + 5)) + 1)*h(i + 1)
        d = {0: np.array([s.subs(j, i).subs(i, k) for k in range(N)], dtype=float),
             1: np.array([s.subs(j, i+1).subs(i, k) for k in range(N-1)], dtype=float),
             2: np.array([s.subs(j, i+2).subs(i, k) for k in range(N-2)], dtype=float),
             3: np.array([s.subs(j, i+3).subs(i, k) for k in range(N-3)], dtype=float)
            }
        #d = {0: (k/(2*k+1))**2*(2/(2*k-1) + 2/(2*k+3)) + ((k+2)/(2*k+3))**2 * (2/(2*k+1)+2/(2*k+5)),
        #     1: 2*k[1:]*(k[1:]+1)/(2*k[1:]+1)**2*(1/(2*k[1:]-1)+1/(2*k[1:]+3)) - 2*(k[1:]+2)*(k[1:]-1)/(2*k[1:]+3)/(2*k[1:]+1)/(2*k[1:]-1),
        #     2: -2*k[2:]*(k[2:]-2)/(2*k[2:]+1)/(2*k[2:]-1)/(2*k[2:]-3)-2*k[2:]*(k[2:]+2)/(2*k[2:]+3)/(2*k[2:]+1)/(2*k[2:]-1),
        #     3: -2*k[3:]*(k[3:]-1)/(2*k[3:]+1)/(2*k[3:]-1)/(2*k[3:]-3)}
        d[-1] = d[1].copy()
        d[-2] = d[2].copy()
        d[-3] = d[3].copy()
        return d

class CUDUDrp1mat(SpectralMatrix):
    r"""Derivative matrix :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj} = (\phi_j, (1+x)\phi'_k)

    where :math:`\phi_k \in` :class:`.legendre.bases.UpperDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], UD)
        assert isinstance(trial[0], UD)
        k = np.arange(test[0].N-1)
        d = {0: -2*(k+1)/(2*k+1)+2*(k+1)/(2*k+3),
             1:  2*(k[1:]+1)/(2*k[1:]+1),
             -1: -2*(k[:-1]+1)/(2*k[:-1]+3)}
        return d


class BUDUDmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj} = (\phi_j, \phi_k)

    where :math:`\phi_k \in` :class:`.legendre.bases.UpperDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], UD)
        assert isinstance(trial[0], UD)
        d0 = get_norm_sq(test[0], trial[0], method)
        d = {0: d0[:-1]+d0[1:], -1: -d0[1:-1]}
        d[1] = d[-1].copy()
        return d

class BLDLDmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj} = (\phi_j, \phi_k)

    where :math:`\phi_k \in` :class:`.legendre.bases.LowerDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], LD)
        assert isinstance(trial[0], LD)
        d0 = get_norm_sq(test[0], trial[0], method)
        d = {0: d0[:-1]+d0[1:], -1: d0[1:-1]}
        d[1] = d[-1].copy()
        return d

class BUDUDrp1mat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj} = (\phi_j, (1+x)\phi_k)

    where :math:`\phi_k \in` :class:`.legendre.bases.UpperDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], UD)
        assert isinstance(trial[0], UD)
        assert test[0].quad == 'LG'
        k = np.arange(test[0].N-1)
        d = {0: 2*k+2}
        d = {0: 4*(k+1)/(2*k+1)/(2*k+3),
             1: 4/(2*k[:-1]+1)/(2*k[:-1]+3)/(2*k[:-1]+5),
             2: -2*(k[:-2]+2)/(2*k[:-2]+3)/(2*k[:-2]+5)}
        d[-1] = d[1].copy()
        d[-2] = d[2].copy()
        return d


class BSDSD1orp1mat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj} = (\phi_j, \frac{1}{1+x}\phi_k)

    where :math:`\phi_k \in` :class:`.legendre.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        assert test[0].quad == 'LG'

        k = np.arange(test[0].N-2)
        d = {0: 2*(2*k+3)/(k+1)/(k+2), 1: -2/(k[:-1]+2), -1: -2/(k[:-1]+2)}
        return d


class BSDSDrp1mat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj} = (\phi_j, (1+x)\phi_k)

    where :math:`\phi_k \in` :class:`.legendre.bases.ShenDirichlet`, and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        #assert test[0].quad == 'LG'

        k = np.arange(test[0].N-2)
        d = {0: 2/(2*k+1)+2/(2*k+5),
             1: 2/(2*k[:-1]+1)/(2*k[:-1]+5) + 2*(k[:-1]+3)/(2*k[:-1]+5)/(2*k[:-1]+7),
             2: -2/(2*k[:-2]+5),
             3: -2*(k[:-3]+3)/(2*k[:-3]+5)/(2*k[:-3]+7)}
        d[-1] = d[1].copy()
        d[-2] = d[2].copy()
        d[-3] = d[3].copy()
        return d


mat = SpectralMatDict({
    ((L,  0), (L,  0)): BLLmat,
    ((L,  0), (L,  1)): CLLmat,
    ((L,  1), (L,  0)): CLLmatT,
    ((L,  0), (SD, 1)): CLSDmat,
    ((SD, 1), (L,  0)): CSDLmat,
    ((SD, 0), (SD, 1)): CSDSDmat,
    ((SD, 1), (SD, 0)): functools.partial(CSDSDmat, scale=-1.),
    ((SD, 0), (SD, 0)): BSDSDmat,
    ((SB, 0), (SB, 0)): BSBSBmat,
    ((SN, 0), (SN, 0)): BSNSNmat,
    ((SD, 0), (L,  0)): BSDLmat,
    ((L,  0), (SD, 0)): BLSDmat,
    ((SD, 1), (SD, 1)): ASDSDmat,
    ((SD, 2), (SD, 0)): functools.partial(ASDSDmat, scale=-1.),
    ((SD, 0), (SD, 2)): functools.partial(ASDSDmat, scale=-1.),
    ((SN, 1), (SN, 1)): ASNSNmat,
    ((SN, 2), (SN, 0)): functools.partial(ASNSNmat, scale=-1.),
    ((SN, 0), (SN, 2)): functools.partial(ASNSNmat, scale=-1.),
    ((L,  2), (L,  0)): ALLmat,
    ((L,  0), (L,  2)): ALLmat,
    ((SB, 2), (SB, 2)): SSBSBmat,
    ((SB, 1), (SB, 1)): ASBSBmat,
    ((SB, 0), (SB, 2)): functools.partial(ASBSBmat, scale=-1.),
    ((SB, 2), (SB, 0)): functools.partial(ASBSBmat, scale=-1.),
    ((SB, 0), (SB, 4)): SSBSBmat,
    ((SB, 4), (SB, 0)): SSBSBmat,
    ((SD, 0), (SD, 2), 1+x): ASDSD2rp1mat,
    ((SD, 2), (SD, 0), 1+x): ASDSD2Trp1mat,
    #((SD, 0), (SD, 0), 1+x): BSDSDrp1mat,
    #((SD, 0), (SD, 0), 1/(1+x)): BSDSD1orp1mat,
    ((UD, 1), (UD, 1), 1+x): AUDUDrp1mat,
    #((UD, 0), (UD, 0), 1+x): BUDUDrp1mat,
    ((UD, 1), (UD, 1), (1+x)**2): AUDUDrp1smat,
    ((UD, 0), (UD, 2), (1+x)**2): GUDUDrp1smat,
    #((UD, 0), (UD, 1), (1+x)): CUDUDrp1mat,
    #((UD, 0), (UD, 0), (1+x)**2): BUDUDrp1smat,
    ((UD, 0), (UD, 0)): BUDUDmat,
    ((LD, 0), (LD, 0)): BLDLDmat,
    ((DN, 0), (DN, 0)): BDNDNmat,
    ((DN, 1), (DN, 1)): ADNDNmat,
    ((DN, 2), (DN, 0)): functools.partial(ADNDNmat, scale=-1.),
    ((DN, 0), (DN, 2)): functools.partial(ADNDNmat, scale=-1.),
    ((BF, 4), (BF, 0)): SBFBFmat,
    ((BF, 0), (BF, 4)): SBFBFmat,
    ((BF, 2), (BF, 2)): SBFBFmat,
    })

#mat = SpectralMatDict({})
