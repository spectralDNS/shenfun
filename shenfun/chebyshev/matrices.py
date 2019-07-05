r"""
This module contains specific inner product matrices for the different bases in
the Chebyshev family.

A naming convention is used for the first three capital letters for all matrices.
The first letter refers to type of matrix.

    - Mass matrices start with `B`
    - One derivative start with `C`
    - Two derivatives (Laplace) start with `A`
    - Four derivatives (Biharmonic) start with `S`

The next two letters refer to the test and trialfunctions, respectively

    - Dirichlet:   `D`
    - Neumann:     `N`
    - Chebyshev:   `T`
    - Biharmonic:  `B`

As such, there are 4 symmetric mass matrices, BDDmat, BNNmat, BTTmat and BBBmat,
corresponding to the four bases above.

A matrix may consist of different types of test and trialfunctions as long as
they are all in the Chebyshev family. A mass matrix using Dirichlet test and
Neumann trial is named BDNmat.

All matrices in this module may be looked up using the 'mat' dictionary,
which takes test and trialfunctions along with the number of derivatives
to be applied to each. As such the mass matrix BDDmat may be looked up
as

>>> from shenfun.chebyshev.matrices import mat
>>> from shenfun.chebyshev.bases import ShenDirichletBasis as SD
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

To see that this is in fact the BDDmat:

>>> print(BM.__class__)
<class 'shenfun.chebyshev.matrices.BDDmat'>

"""
#pylint: disable=bad-continuation, redefined-builtin

from __future__ import division

#__all__ = ['mat']

import numpy as np
from shenfun.optimization import cython
from shenfun.matrixbase import SpectralMatrix
from shenfun.utilities import inheritdocstrings
from shenfun.la import TDMA as neumann_TDMA
from .la import TDMA
from . import bases

# Short names for instances of bases
CB = bases.Basis
SD = bases.ShenDirichletBasis
SB = bases.ShenBiharmonicBasis
SN = bases.ShenNeumannBasis

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


@inheritdocstrings
class BDDmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        B_{kj}=(\phi_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\phi_j` is a Shen Dirichlet basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        ck = get_ck(test[0].N, test[0].quad)
        d = {0: np.pi/2*(ck[:-2]+ck[2:]),
             2: np.array([-np.pi/2])}
        d[-2] = d[2]
        SpectralMatrix.__init__(self, d, test, trial)
        self.solve = TDMA(self)

    def matvec(self, v, c, format='cython', axis=0):
        N, M = self.shape
        c.fill(0)
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
            s = (slice(None),)+(np.newaxis,)*(v.ndim-1) # broadcasting
            c[:(N-2)] = self[2]*v[2:N]
            c[:N] += self[0][s]*v[:N]
            c[2:N] += self[-2]*v[:(N-2)]
            if axis > 0:
                v = np.moveaxis(v, 0, axis)
                c = np.moveaxis(c, 0, axis)
            self.scale_array(c)

        else:
            c = super(BDDmat, self).matvec(v, c, format=format, axis=axis)

        return c


@inheritdocstrings
class BNDmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\phi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 1, 2, ..., N-2

    :math:`\psi_k` is the Shen Dirichlet basis function and :math:`\phi_j` is a
    Shen Neumann basis function.

    For simplicity, the matrix is stored including the zero index row
    (:math:`k=0`)
    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SD)
        N = test[0].N
        ck = get_ck(N, test[0].quad)
        k = np.arange(N-2, dtype=np.float)
        d = {-2: -np.pi/2,
              0: np.pi/2.*(ck[:-2]+ck[2:]*(k/(k+2))**2),
              2: -np.pi/2*(k[:N-4]/(k[:N-4]+2))**2}
        SpectralMatrix.__init__(self, d, test, trial)

    def matvec(self, v, c, format='csr', axis=0):
        c = super(BNDmat, self).matvec(v, c, format=format, axis=axis)
        s = [slice(None),]*v.ndim
        s[axis] = 0
        c[tuple(s)] = 0
        return c


@inheritdocstrings
class BDNmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\psi_j, \phi_k)_w

    where

    .. math::

        j = 1, 2, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    :math:`\psi_j` is the Shen Dirichlet basis function and :math:`\phi_k` is a
    Shen Neumann basis function.

    For simplicity, the matrix is stored including the zero index column
    (:math:`j=0`)
    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SN)
        N = test[0].N
        ck = get_ck(N, test[0].quad)
        k = np.arange(N-2, dtype=np.float)
        d = {-2: -np.pi/2*(k[:N-4]/(k[:N-4]+2))**2,
              0:  np.pi/2.*(ck[:-2]+ck[2:]*(k/(k+2))**2),
              2: -np.pi/2}
        SpectralMatrix.__init__(self, d, test, trial)

    def matvec(self, v, c, format='cython', axis=0):
        c.fill(0)
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
            c = super(BDNmat, self).matvec(v, c, format=format, axis=axis)

        return c


@inheritdocstrings
class BNTmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (T_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 1, 2, ..., N-2

    :math:`\phi_k` is the Shen Neumann basis function and :math:`T_j` is a
    Chebyshev basis function.

    For simplicity, the matrix is stored including the zero index row
    (:math:`k=0`)

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], CB)
        SpectralMatrix.__init__(self, {}, test, trial)

    def matvec(self, v, c, format='csr', axis=0):
        c = super(BNTmat, self).matvec(v, c, format=format, axis=axis)
        s = [slice(None),]*v.ndim
        s[axis] = 0
        c[tuple(s)] = 0
        return c


@inheritdocstrings
class BNBmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\phi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-4 \text{ and } k = 1, 2, ..., N-2

    :math:`\psi_k` is the Shen Neumann basis function and :math:`\phi_j` is a
    Shen biharmonic basis function.

    For simplicity, the matrix is stored including the zero index row
    (:math:`k=0`)

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SB)
        SpectralMatrix.__init__(self, {}, test, trial)

    def matvec(self, v, c, format='csr', axis=0):
        c = super(BNBmat, self).matvec(v, c, format=format, axis=axis)
        s = [slice(None),]*v.ndim
        s[axis] = 0
        c[tuple(s)] = 0
        return c


@inheritdocstrings
class BTTmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (T_j, T_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N

    and :math:`T_j` is the jth order Chebyshev function of the first kind.
    """
    def __init__(self, test, trial):
        assert isinstance(test[0], CB)
        assert isinstance(trial[0], CB)
        ck = get_ck(test[0].N, test[0].quad)
        SpectralMatrix.__init__(self, {0: np.pi/2*ck}, test, trial)

    def matvec(self, v, c, format='self', axis=0):
        c.fill(0)
        if format == 'self':
            s = [np.newaxis,]*v.ndim # broadcasting
            s[axis] = slice(None)
            c[:] = self[0][tuple(s)]*v
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

@inheritdocstrings
class BNNmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\phi_j, \phi_k)_w

    where

    .. math::

        j = 1, 2, ..., N-2 \text{ and } k = 1, 2, ..., N-2

    and :math:`\phi_j` is the Shen Neumann basis function.

    The matrix is stored including the zero index row and column

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SN)
        N = test[0].N
        ck = get_ck(N, test[0].quad)
        k = np.arange(N-2, dtype=np.float)
        d = {0: np.pi/2*(ck[:-2]+ck[2:]*(k[:]/(k[:]+2))**4),
             2: -np.pi/2*((k[2:]-2)/(k[2:]))**2}
        d[-2] = d[2]
        SpectralMatrix.__init__(self, d, test, trial)
        self.solve = neumann_TDMA(self)

    def matvec(self, v, c, format='csr', axis=0):
        c = super(BNNmat, self).matvec(v, c, format=format, axis=axis)
        s = [slice(None),]*v.ndim
        s[axis] = 0
        c[tuple(s)] = 0
        return c


@inheritdocstrings
class BDTmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (T_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N-2

    :math:`\phi_k` is the Shen Dirichlet basis function and :math:`T_j` is the
    Chebyshev basis function.
    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], CB)
        N = test[0].N
        ck = get_ck(N, test[0].quad)
        d = {0: np.pi/2*ck[:N-2],
             2: -np.pi/2*ck[2:]}
        SpectralMatrix.__init__(self, d, test, trial)


@inheritdocstrings
class BTDmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\phi_j, T_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N

    :math:`\phi_j` is the Shen Dirichlet basis function and :math:`T_k` is the
    Chebyshev basis function.
    """
    def __init__(self, test, trial):
        assert isinstance(test[0], CB)
        assert isinstance(trial[0], SD)
        N = test[0].N
        ck = get_ck(N, test[0].quad)
        d = {-2: -np.pi/2*ck[2:],
              0: np.pi/2*ck[:N-2]}
        SpectralMatrix.__init__(self, d, test, trial)


@inheritdocstrings
class BTNmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\phi_j, T_k)_w

    where

    .. math::

        j = 1, 2, ..., N-2 \text{ and } k = 0, 1, ..., N

    :math:`\phi_j` is the Shen Neumann basis function and :math:`T_k` is the
    Chebyshev basis function.
    """
    def __init__(self, test, trial):
        assert isinstance(test[0], CB)
        assert isinstance(trial[0], SN)
        N = test[0].N
        ck = get_ck(N, test[0].quad)
        k = np.arange(N, dtype=np.float)
        d = {-2: -np.pi/2*ck[2:]*((k[2:]-2)/k[2:])**2,
              0: np.pi/2*ck[:-2]}
        SpectralMatrix.__init__(self, d, test, trial)


@inheritdocstrings
class BBBmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\psi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-4 \text{ and } k = 0, 1, ..., N-4

    and :math:`\psi_j` is the Shen Biharmonic basis function.
    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        from shenfun.la import PDMA
        N = test[0].N
        ck = get_ck(N, test[0].quad)
        k = np.arange(N-4, dtype=np.float)
        d = {4: (k[:-4]+1)/(k[:-4]+3)*np.pi/2,
             2: -((k[:-2]+2)/(k[:-2]+3) + (k[:-2]+4)*(k[:-2]+1)/((k[:-2]+5)*(k[:-2]+3)))*np.pi,
             0: (ck[:N-4] + 4*((k+2)/(k+3))**2 + ck[4:]*((k+1)/(k+3))**2)*np.pi/2.}
        d[-2] = d[2]
        d[-4] = d[4]
        SpectralMatrix.__init__(self, d, test, trial)
        self.solve = PDMA(self)

    def matvec(self, v, c, format='cython', axis=0):
        c.fill(0)
        N = self.shape[0]
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
            c = super(BBBmat, self).matvec(v, c, format=format, axis=axis)

        return c


@inheritdocstrings
class BBDmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\phi_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-4

    and :math:`\phi_j` is the Shen Dirichlet basis function and :math:`\psi_k`
    the Shen Biharmonic basis function.
    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SD)
        N = test[0].N
        ck = get_ck(N, test[0].quad)
        k = np.arange(N-4, dtype=np.float)
        a = 2*(k+2)/(k+3)
        b = (k[:N-4]+1)/(k[:N-4]+3)
        d = {-2: -np.pi/2,
              0: (ck[:N-4] + a)*np.pi/2,
              2: -(a+b*ck[4:])*np.pi/2,
              4: b[:-2]*np.pi/2}
        SpectralMatrix.__init__(self, d, test, trial)

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
            c = super(BBDmat, self).matvec(v, c, format=format, axis=axis)

        return c

@inheritdocstrings
class CDNmat(SpectralMatrix):
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
    def __init__(self, test, trial):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SN)
        N = test[0].N
        k = np.arange(N-2, dtype=np.float)
        d = {-1: -((k[1:]-1)/(k[1:]+1))**2*(k[1:]+1)*np.pi,
              1: (k[:-1]+1)*np.pi}
        SpectralMatrix.__init__(self, d, test, trial)

    def matvec(self, v, c, format='cython', axis=0):
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
            c = super(CDNmat, self).matvec(v, c, format=format, axis=axis)

        return c


@inheritdocstrings
class CDDmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (\phi'_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\phi_k` is the Shen Dirichlet basis function.
    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=np.float)
        d = {-1: -(k[1:N-2]+1)*np.pi,
              1: (k[:(N-3)]+1)*np.pi}
        SpectralMatrix.__init__(self, d, test, trial)

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
            c = super(CDDmat, self).matvec(v, c, format=format, axis=axis)

        return c


@inheritdocstrings
class CNDmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (\phi'_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 1, 2, ..., N-2

    and :math:`\phi_j` is the Shen Dirichlet basis function and :math:`\psi_k`
    the Shen Neumann basis function.

    For simplicity, the matrix is stored including the zero index coloumn (j=0)

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N-2, dtype=np.float)
        d = {-1: -(k[1:]+1)*np.pi,
              1: -(2-k[:-1]**2/(k[:-1]+2)**2*(k[:-1]+3))*np.pi}
        for i in range(3, N-1, 2):
            d[i] = -(1-k[:-i]**2/(k[:-i]+2)**2)*2*np.pi
        SpectralMatrix.__init__(self, d, test, trial)

    def matvec(self, v, c, format='csr', axis=0):
        c = super(CNDmat, self).matvec(v, c, format=format, axis=axis)
        s = [slice(None),]*v.ndim
        s[axis] = 0
        c[tuple(s)] = 0
        return c


@inheritdocstrings
class CTDmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (\phi'_j, T_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N

    :math:`\phi_j` is the Shen Dirichlet basis function and :math:`T_k` is the
    Chebyshev basis function.
    """
    def __init__(self, test, trial):
        assert isinstance(test[0], CB)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=np.float)
        d = {-1: -(k[1:N-1]+1)*np.pi,
              1: -2*np.pi}
        for i in range(3, N-2, 2):
            d[i] = -2*np.pi
        SpectralMatrix.__init__(self, d, test, trial)


class CDTmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (T'_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N-2

    :math:`\phi_k` is the Shen Dirichlet basis function and :math:`T_j` is the
    Chebyshev basis function.
    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], CB)
        N = test[0].N
        k = np.arange(N, dtype=np.float)
        d = {1: np.pi*(k[:N-2]+1)}
        SpectralMatrix.__init__(self, d, test, trial)


@inheritdocstrings
class CBDmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (\phi'_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-4

    :math:`\phi_j` is the Shen Dirichlet basis and :math:`\psi_k` the Shen
    Biharmonic basis function.
    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=np.float)
        d = {-1: -(k[1:N-4]+1)*np.pi,
              1: 2*(k[:N-4]+1)*np.pi,
              3: -(k[:N-5]+1)*np.pi}
        SpectralMatrix.__init__(self, d, test, trial)

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
            c = super(CBDmat, self).matvec(v, c, format=format, axis=axis)
        return c


@inheritdocstrings
class CDBmat(SpectralMatrix):
    r"""Matrix for inner product

    .. math::

        C_{kj} = (\psi'_j, \phi_k)_w

    where

        j = 0, 1, ..., N-4 \text{ and } k = 0, 1, ..., N-2

    :math:`\phi_k` is the Shen Dirichlet basis function and :math:`\psi_j` the
    Shen Biharmonic basis function.
    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SB)
        N = test[0].N
        k = np.arange(N, dtype=np.float)
        d = {-3: (k[3:-2]-2)*(k[3:-2]+1)/k[3:-2]*np.pi,
             -1: -2*(k[1:-3]+1)**2/(k[1:-3]+2)*np.pi,
              1: (k[:-5]+1)*np.pi}
        SpectralMatrix.__init__(self, d, test, trial)

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
            c = super(CDBmat, self).matvec(v, c, format=format, axis=axis)

        return c


@inheritdocstrings
class ABBmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\psi''_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-4 \text{ and } k = 0, 1, ..., N-4

    and :math:`\psi_k` is the Shen Biharmonic basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        N = test[0].N
        k = np.arange(N-4, dtype=np.float)
        d = {-2: 2*(k[2:]-1)*(k[2:]+2)*np.pi,
              0: -4*((k+1)*(k+2)**2)/(k+3)*np.pi,
              2: 2*(k[:-2]+1)*(k[:-2]+2)*np.pi}
        SpectralMatrix.__init__(self, d, test, trial)

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
            c = super(ABBmat, self).matvec(v, c, format=format, axis=axis)

        return c


@inheritdocstrings
class ADDmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\psi''_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-2 \text{ and } k = 0, 1, ..., N-2

    and :math:`\psi_k` is the Shen Dirichlet basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SD)
        N = test[0].N
        k = np.arange(N, dtype=np.float)
        d = {0: -2*np.pi*(k[:N-2]+1)*(k[:N-2]+2)}
        for i in range(2, N-2, 2):
            d[i] = -4*np.pi*(k[:-(i+2)]+1)
        SpectralMatrix.__init__(self, d, test, trial)

        # Following storage more efficient, but requires effort in iadd/isub...
        #d = {0: -2*np.pi*(k[:N-2]+1)*(k[:N-2]+2),
             #2: -4*np.pi*(k[:-4]+1)}
        #for i in range(4, N-2, 2):
            #d[i] = d[2][:2-i]
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
            c = super(ADDmat, self).matvec(v, c, format=format, axis=axis)

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
            if not u is b:
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
        self.testfunction[0].bc.apply_after(u, True)
        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if not u is b:
                b = np.moveaxis(b, 0, axis)
        return u


@inheritdocstrings
class ANNmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\phi''_j, \phi_k)_w

    where

    .. math::

        j = 1, 2, ..., N-2 \text{ and } k = 1, 2, ..., N-2

    and :math:`\phi_k` is the Shen Neumann basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SN)
        N = test[0].N
        k = np.arange(N-2, dtype=np.float)
        d = {0: -2*np.pi*k**2*(k+1)/(k+2)}
        for i in range(2, N-2, 2):
            d[i] = -4*np.pi*(k[:-i]+i)**2*(k[:-i]+1)/(k[:-i]+2)**2
        SpectralMatrix.__init__(self, d, test, trial)

    def matvec(self, v, c, format='csr', axis=0):
        c = super(ANNmat, self).matvec(v, c, format=format, axis=axis)
        s = [slice(None),]*v.ndim
        s[axis] = 0
        c[tuple(s)] = self.testfunction[0].mean*np.pi
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
            if not u is b:
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
        us[0] = self.testfunction[0].mean
        sl = [np.newaxis]*b.ndim
        sl[0] = slice(None)
        us *= j2[tuple(sl)]

        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if not u is b:
                b = np.moveaxis(b, 0, axis)
        u /= self.scale
        return u


@inheritdocstrings
class ATTmat(SpectralMatrix):
    r"""Stiffness matrix for inner product

    .. math::

        A_{kj} = (\psi''_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N

    and :math:`\psi_k` is the Chebyshev basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], CB)
        assert isinstance(trial[0], CB)
        N = test[0].N
        k = np.arange(N, dtype=np.float)
        d = {}
        for j in range(2, N, 2):
            d[j] = k[j:]*(k[j:]**2-k[:-j]**2)*np.pi/2.
        SpectralMatrix.__init__(self, d, test, trial)


@inheritdocstrings
class SBBmat(SpectralMatrix):
    r"""Biharmonic matrix for inner product

    .. math::

        S_{kj} = (\psi''''_j, \psi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-4 \text{ and } k = 0, 1, ..., N-4

    and :math:`\psi_k` is the Shen Biharmonic basis function.
    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SB)
        assert isinstance(trial[0], SB)
        N = test[0].N
        ki = np.arange(N-4)
        k = np.arange(N-4, dtype=np.float)
        i = 8*(ki+1)**2*(ki+2)*(ki+4)
        d = {0: i*np.pi}
        for j in range(2, N-4, 2):
            i = 8*(ki[:-j]+1)*(ki[:-j]+2)*(ki[:-j]*(ki[:-j]+4)+3*(ki[j:]+2)**2)
            d[j] = np.array(i*np.pi/(k[j:]+3))
        SpectralMatrix.__init__(self, d, test, trial)

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
            c = super(SBBmat, self).matvec(v, c, format=format, axis=axis)

        return c


@inheritdocstrings
class _Chebmatrix(SpectralMatrix):
    def __init__(self, test, trial):
        SpectralMatrix.__init__(self, {}, test, trial)


class _ChebMatDict(dict):
    """Dictionary of inner product matrices

    Matrices that are missing keys are generated from Vandermonde type
    computations.

    """

    def __missing__(self, key):
        c = _Chebmatrix
        self[key] = c
        return c

    def __getitem__(self, key):
        matrix = dict.__getitem__(self, key)
        assert key[0][1] == 0, 'Test cannot be differentiated (weighted space)'
        return matrix


# Define dictionary to hold all predefined matrices
# When looked up, missing matrices will be generated automatically
mat = _ChebMatDict({
    ((CB, 0), (CB, 0)): BTTmat,
    ((SD, 0), (SD, 0)): BDDmat,
    ((SB, 0), (SB, 0)): BBBmat,
    ((SN, 0), (SN, 0)): BNNmat,
    ((SN, 0), (CB, 0)): BNTmat,
    ((SN, 0), (SB, 0)): BNBmat,
    ((SD, 0), (SN, 0)): BDNmat,
    ((SN, 0), (SD, 0)): BNDmat,
    ((CB, 0), (SN, 0)): BTNmat,
    ((SB, 0), (SD, 0)): BBDmat,
    ((CB, 0), (SD, 0)): BTDmat,
    ((SD, 0), (CB, 0)): BDTmat,
    ((SD, 0), (SD, 2)): ADDmat,
    ((CB, 0), (CB, 2)): ATTmat,
    ((SN, 0), (SN, 2)): ANNmat,
    ((SB, 0), (SB, 2)): ABBmat,
    ((SB, 0), (SB, 4)): SBBmat,
    ((SD, 0), (SN, 1)): CDNmat,
    ((SB, 0), (SD, 1)): CBDmat,
    ((CB, 0), (SD, 1)): CTDmat,
    ((SD, 0), (SD, 1)): CDDmat,
    ((SN, 0), (SD, 1)): CNDmat,
    ((SD, 0), (SB, 1)): CDBmat,
    ((SD, 0), (CB, 1)): CDTmat
    })
