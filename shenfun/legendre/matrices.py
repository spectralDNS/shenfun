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
>>> from shenfun.legendre.bases import ShenDirichletBasis as SD
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

import numpy as np
from shenfun.matrixbase import SpectralMatrix
from shenfun.utilities import inheritdocstrings
from shenfun.la import TDMA as neumann_TDMA
from .la import TDMA
from . import bases

# Short names for instances of bases
LB = bases.Basis
SD = bases.ShenDirichletBasis
SB = bases.ShenBiharmonicBasis
SN = bases.ShenNeumannBasis

#pylint: disable=unused-variable, redefined-builtin, bad-continuation

@inheritdocstrings
class BLLmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (L_j, L_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N

    and :math:`L_k` is the Legendre basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], LB)
        assert isinstance(trial[0], LB)
        N = test[0].N
        k = np.arange(N, dtype=np.float)
        d = {0: 2./(2.*k+1)}
        if test[0].quad == 'GL':
            d[0][-1] = 2./(N-1)
        SpectralMatrix.__init__(self, d, test, trial)

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


@inheritdocstrings
class BDDmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\psi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial):
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
        SpectralMatrix.__init__(self, d, test, trial)
        self.solve = TDMA(self)


@inheritdocstrings
class BNNmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\psi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Neumann basis function.

    """
    def __init__(self, test, trial):
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
        SpectralMatrix.__init__(self, d, test, trial)
        #self.solve = neumann_TDMA(self)


@inheritdocstrings
class BBBmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\psi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Biharmonic basis function.

    """
    def __init__(self, test, trial):
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
        SpectralMatrix.__init__(self, d, test, trial)
        self.solve = PDMA(self)

@inheritdocstrings
class BDLmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (L_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial):
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
        SpectralMatrix.__init__(self, d, test, trial)

@inheritdocstrings
class BLDmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\psi_j, L_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N

    and :math:`\psi_j` is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial):
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
        SpectralMatrix.__init__(self, d, test, trial)

@inheritdocstrings
class ADDmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\psi'_j, \psi'_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N-2, dtype=np.float)
        if not test[0].is_scaled():
            d = {0: 4*k+6}
        else:
            d = {0: 1}
        SpectralMatrix.__init__(self, d, test, trial)

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
                if not u is b:
                    b = np.moveaxis(b, axis, 0)

            bs = b[s]
            us = u[s]
            d = 1./self[0]
            sl = [np.newaxis]*bs.ndim
            sl[0] = slice(None)
            us[:] = bs*d[tuple(sl)]
            u /= self.scale
            self.testfunction[0].bc.apply_after(u, True)

            if axis > 0:
                u = np.moveaxis(u, 0, axis)
                if not u is b:
                    b = np.moveaxis(b, axis, 0)
        else:
            ss = [slice(None)]*b.ndim
            ss[axis] = s
            ss = tuple(ss)
            u[ss] = b[ss]
            u /= self.scale
            self.testfunction[0].bc.apply_after(u, True)

        return u

@inheritdocstrings
class GDDmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        G_{kj} = (\psi''_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N-2, dtype=np.float)
        if not test[0].is_scaled():
            d = {0: -(4*k+6)}
        else:
            d = {0: -1}
        SpectralMatrix.__init__(self, d, test, trial)

    def solve(self, b, u=None, axis=0):
        N = self.shape[0] + 2
        assert N == b.shape[0]
        s = self.trialfunction[0].slice()

        if u is None:
            u = b
        else:
            assert u.shape == b.shape

        if not self.trialfunction[0].is_scaled():

            # Move axis to first
            if axis > 0:
                u = np.moveaxis(u, axis, 0)
                if not u is b:
                    b = np.moveaxis(b, axis, 0)

            bs = b[s]
            us = u[s]
            d = 1./self[0]
            sl = [np.newaxis]*bs.ndim
            sl[0] = slice(None)
            us[:] = bs*d[tuple(sl)]
            self.testfunction[0].bc.apply_after(u, True)

            if axis > 0:
                u = np.moveaxis(u, 0, axis)
                if not u is b:
                    b = np.moveaxis(b, axis, 0)
        else:
            ss = [slice(None)]*b.ndim
            ss[axis] = s
            u[ss] = -b[tuple(ss)]
            self.testfunction[0].bc.apply_after(u, True)

        u /= self.scale
        return u


@inheritdocstrings
class ANNmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\psi'_j, \psi'_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Neumann basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SN)
        N = test[0].N
        k = np.arange(N-2, dtype=np.float)
        alpha = k*(k+1)/(k+2)/(k+3)
        d0 = 2./(2*k+1)
        d = {0: d0*alpha*(k+0.5)*((k+2)*(k+3)-k*(k+1))}
        SpectralMatrix.__init__(self, d, test, trial)

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
            if not u is b:
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
            if not u is b:
                b = np.moveaxis(b, axis, 0)

        u /= self.scale
        return u


@inheritdocstrings
class ABBmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\psi'_j, \psi'_k)_w

    where

    .. math::

        j = 0, 1, ..., N-4 \text{ and } k = 0, 1, ..., N-4

    and :math:`\psi_k` is the Shen Legendre Biharmonic basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        N = test[0].N
        k = np.arange(N-4, dtype=np.float)
        gk = (2*k+3)/(2*k+7)
        d = {0: 2*(2*k+3)*(1+gk),
             2: -2*(2*k[:-2]+3)}
        d[-2] = d[2]
        SpectralMatrix.__init__(self, d, test, trial)


@inheritdocstrings
class GLLmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        B_{kj} = (L_j'', L_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N

    and :math:`L_k` is the Legendre basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], LB)
        assert isinstance(trial[0], LB)
        N = test[0].N
        k = np.arange(N, dtype=np.float)
        d = {}
        for j in range(2, N, 2):
            jj = j if trial[1] else -j
            d[jj] = (k[:-j]+0.5)*((k[:-j]+j)*(k[:-j]+j+1) - k[:-j]*(k[:-j]+1))*2./(2*k[:-j]+1)
        SpectralMatrix.__init__(self, d, test, trial)


@inheritdocstrings
class PBBmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\psi_j, \psi''_k)_w

    where

    .. math::

        j = 0, 1, ..., N-4 \text{ and } k = 0, 1, ..., N-4

    and :math:`\psi_k` is the Shen Legendre Biharmonic basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        N = test[0].N
        k = np.arange(N-4, dtype=np.float)
        gk = (2*k+3)/(2*k+7)
        d = {0: -2*(2*k+3)*(1+gk),
             2: 2*(2*k[:-2]+3)}
        d[-2] = d[2]
        SpectralMatrix.__init__(self, d, test, trial)


@inheritdocstrings
class SBBmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\psi''_j, \psi''_k)_w

    where

    .. math::

        j = 0, 1, ..., N-4 \text{ and } k = 0, 1, ..., N-4

    and :math:`\psi_k` is the Shen Legendre Biharmonic basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        N = test[0].N
        k = np.arange(N-4, dtype=np.float)
        d = {0: 2*(2*k+3)**2*(2*k+5)}
        SpectralMatrix.__init__(self, d, test, trial)

@inheritdocstrings
class CLLmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (\psi'_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N

    and :math:`\psi_k` is the Shen Legendre basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], LB)
        assert isinstance(trial[0], LB)
        N = test[0].N
        d = {}
        for i in range(1, N, 2):
            d[i] = 2.0
        SpectralMatrix.__init__(self, d, test, trial)

    def matvec(self, v, c, format='self', axis=0):
        N = self.shape[0]
        c.fill(0)
        if format == 'self':
            if axis > 0:
                c = np.moveaxis(c, axis, 0)
                v = np.moveaxis(v, axis, 0)
            s = (slice(None),) + (np.newaxis,)*(v.ndim-1) # broadcasting
            ve = v[-2:0:-2].cumsum(axis=0)
            vo = v[-1:0:-2].cumsum(axis=0)
            c[-3::-2] = ve*2.0
            c[-2::-2] = vo*2.0
            if axis > 0:
                c = np.moveaxis(c, 0, axis)
                v = np.moveaxis(v, 0, axis)
            c *= self.scale

        else:
            c = super(CLLmat, self).matvec(v, c, format=format, axis=axis)

        return c


@inheritdocstrings
class CLLmatT(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (\psi'_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N

    and :math:`\psi_k` is the Shen Legendre basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], LB)
        assert isinstance(trial[0], LB)
        N = test[0].N
        d = {}
        for i in range(-1, -N, -2):
            d[i] = 2.0
        SpectralMatrix.__init__(self, d, test, trial)

@inheritdocstrings
class CLDmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (\psi'_j, L_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N

    and :math:`\psi_k` is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], LB)
        assert isinstance(trial[0], SD)
        N = test[0].N
        d = {-1: -2}
        if trial[0].is_scaled():
            k = np.arange(N-2, dtype=np.float)
            d[-1] = -2. / np.sqrt(4*k+6)
        SpectralMatrix.__init__(self, d, test, trial)

@inheritdocstrings
class CDLmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (L_j, \psi'_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Legendre Dirichlet basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], LB)
        N = test[0].N
        d = {1: -2}
        if test[0].is_scaled():
            k = np.arange(N-2, dtype=np.float)
            d[1] = -2. / np.sqrt(4*k+6)
        SpectralMatrix.__init__(self, d, test, trial)


@inheritdocstrings
class _Legmatrix(SpectralMatrix):
    def __init__(self, test, trial):
        SpectralMatrix.__init__(self, {}, test, trial)


class _LegMatDict(dict):
    """Dictionary of inner product matrices

    Matrices that are missing keys are generated from Vandermonde type
    computations.

    """

    def __missing__(self, key):
        c = _Legmatrix
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
    ((SD, 0), (SD, 0)): BDDmat,
    ((SB, 0), (SB, 0)): BBBmat,
    ((SN, 0), (SN, 0)): BNNmat,
    ((SD, 0), (LB, 0)): BDLmat,
    ((LB, 0), (SD, 0)): BLDmat,
    ((SD, 1), (SD, 1)): ADDmat,
    ((SD, 2), (SD, 0)): GDDmat,
    ((SD, 0), (SD, 2)): GDDmat,
    ((SN, 1), (SN, 1)): ANNmat,
    ((LB, 2), (LB, 0)): GLLmat,
    ((LB, 0), (LB, 2)): GLLmat,
    ((SB, 2), (SB, 2)): SBBmat,
    ((SB, 1), (SB, 1)): ABBmat,
    ((SB, 0), (SB, 2)): PBBmat,
    ((SB, 2), (SB, 0)): PBBmat,
    ((SB, 0), (SB, 4)): SBBmat,
    ((SB, 4), (SB, 0)): SBBmat
    })
