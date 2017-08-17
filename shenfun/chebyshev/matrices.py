from __future__ import division

__all__ = ['mat']

import numpy as np
from shenfun.optimization.Matvec import CDNmat_matvec, BDNmat_matvec, \
    CDDmat_matvec, SBBmat_matvec, SBBmat_matvec3D, Tridiagonal_matvec, \
    Tridiagonal_matvec3D, Pentadiagonal_matvec, Pentadiagonal_matvec3D, \
    CBD_matvec3D, CBD_matvec, CDB_matvec3D, ADDmat_matvec, BBD_matvec3D

from shenfun.matrixbase import SpectralMatrix
from shenfun.utilities import inheritdocstrings
from .la import TDMA
from . import bases

# Short names for instances of bases
CB = bases.Basis
SD = bases.ShenDirichletBasis
SB = bases.ShenBiharmonicBasis
SN = bases.ShenNeumannBasis

def get_ck(N, quad):
    ck = np.ones(N, int)
    ck[0] = 2
    if quad == "GL": ck[-1] = 2
    return ck


@inheritdocstrings
class BDDmat(SpectralMatrix):
    """Matrix for inner product B_{kj}=(phi_j, phi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-2

    and phi_j is a Shen Dirichlet basis function.

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
            Tridiagonal_matvec3D(v, c, ld, self[0], ld, axis)

        elif format == 'cython' and v.ndim == 1:
            ld = self[-2]*np.ones(M-2)
            Tridiagonal_matvec(v, c, ld, self[0], ld)

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

        else:
            c = super(BDDmat, self).matvec(v, c, format=format, axis=axis)

        return c


@inheritdocstrings
class BNDmat(SpectralMatrix):
    """Mass matrix for inner product B_{kj} = (phi_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 1, 2, ..., N-2

    psi_k is the Shen Dirichlet basis function and phi_j is a Shen Neumann
    basis function.

    For simplicity, the matrix is stored including the zero index row (k=0)

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
        c = super(SpectralMatrix, self).matvec(v, c, format=format, axis=axis)
        s = [slice(None),]*v.ndim
        s[axis] = 0
        c[s] = 0
        return c


@inheritdocstrings
class BDNmat(SpectralMatrix):
    """Mass matrix for inner product B_{kj} = (psi_j, phi_k)_w

    where

        j = 1, 2, ..., N-2 and k = 0, 1, ..., N-2

    psi_j is the Shen Dirichlet basis function and phi_k is a Shen Neumann
    basis function.

    For simplicity, the matrix is stored including the zero index column (j=0)

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
            BDNmat_matvec(self[2], self[-2], self[0], v, c, axis)

        else:
            c = super(BDNmat, self).matvec(v, c, format=format, axis=axis)

        return c

@inheritdocstrings
class BNTmat(SpectralMatrix):
    """Mass matrix for inner product B_{kj} = (phi_j, psi_k)_w

    where

        j = 0, 1, ..., N and k = 1, 2, ..., N-2

    psi_k is the Shen Neumann basis function and phi_j is a Chebyshev
    basis function.

    For simplicity, the matrix is stored including the zero index row (k=0)

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], CB)
        SpectralMatrix.__init__(self, {}, test, trial)

    def matvec(self, v, c, format='csr', axis=0):
        c = super(SpectralMatrix, self).matvec(v, c, format=format, axis=axis)
        s = [slice(None),]*v.ndim
        s[axis] = 0
        c[s] = 0
        return c


@inheritdocstrings
class BNBmat(SpectralMatrix):
    """Mass matrix for inner product B_{kj} = (phi_j, psi_k)_w

    where

        j = 0, 1, ..., N-4 and k = 1, 2, ..., N-2

    psi_k is the Shen Neumann basis function and phi_j is a Shen biharmonic
    basis function.

    For simplicity, the matrix is stored including the zero index row (k=0)

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SB)
        SpectralMatrix.__init__(self, {}, test, trial)

    def matvec(self, v, c, format='csr', axis=0):
        c = super(SpectralMatrix, self).matvec(v, c, format=format, axis=axis)
        s = [slice(None),]*v.ndim
        s[axis] = 0
        c[s] = 0
        return c


@inheritdocstrings
class BTTmat(SpectralMatrix):
    """Mass matrix for inner product B_{kj} = (T_j, T_k)_w

    where

        j = 0, 1, ..., N and k = 0, 1, ..., N

    and T_j is the jth order Chebyshev function of the first kind.

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
            c[:] = self[0][s]*v
        else:
            c = super(SpectralMatrix, self).matvec(v, c, format=format, axis=axis)

        return c


@inheritdocstrings
class BNNmat(SpectralMatrix):
    """Mass matrix for inner product B_{kj} = (phi_j, phi_k)_w

    where

        j = 1, 2, ..., N-2 and k = 1, 2, ..., N-2

    and phi_j is the Shen Neumann basis function.

    The matrix is stored including the zero index row and column

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SN)
        assert isinstance(trial[0], SN)
        from shenfun.la import TDMA
        N = test[0].N
        ck = get_ck(N, test[0].quad)
        k = np.arange(N-2, dtype=np.float)
        d = {0: np.pi/2*(ck[:-2]+ck[2:]*(k[:]/(k[:]+2))**4),
             2: -np.pi/2*((k[2:]-2)/(k[2:]))**2}
        d[-2] = d[2]
        SpectralMatrix.__init__(self, d, test, trial)
        self.solve = TDMA(self)

    def matvec(self, v, c, format='csr', axis=0):
        c = super(SpectralMatrix, self).matvec(v, c, format=format, axis=axis)
        s = [slice(None),]*v.ndim
        s[axis] = 0
        c[s] = 0
        return c


@inheritdocstrings
class BDTmat(SpectralMatrix):
    """Mass matrix for inner product B_{kj} = (T_j, phi_k)_w

    where

        j = 0, 1, ..., N and k = 0, 1, ..., N-2

    phi_k is the Shen Dirichlet basis function and T_j is the Chebyshev basis
    function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], CB)
        N = test[0].N
        ck = get_ck(N, test[0].quad)
        k = np.arange(N-2, dtype=np.float)
        d = {0: np.pi/2*ck[:N-2],
             2: -np.pi/2*ck[2:]}
        SpectralMatrix.__init__(self, d, test, trial)


@inheritdocstrings
class BTDmat(SpectralMatrix):
    """Mass matrix for inner product B_{kj} = (phi_j, T_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N

    phi_j is the Shen Dirichlet basis function and T_k is the Chebyshev basis
    function.

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
    """Mass matrix for inner product B_{kj} = (phi_j, T_k)_w

    where

        j = 1, 2, ..., N-2 and k = 0, 1, ..., N

    phi_J is the Shen Neumann basis function and T_k is the Chebyshev basis
    function.

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
    """Mass matrix for inner product B_{kj} = (psi_j, psi_k)_w

    where

        j = 0, 1, ..., N-4 and k = 0, 1, ..., N-4

    and psi_j is the Shen Biharmonic basis function.

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

        elif format == 'cython' and v.ndim == 3:
            Pentadiagonal_matvec3D(v, c, self[-4], self[-2], self[0],
                                   self[2], self[4], axis)

        elif format == 'cython' and v.ndim == 1:
            Pentadiagonal_matvec(v, c, self[-4], self[-2], self[0],
                                 self[2], self[4])
        else:
            c = super(SpectralMatrix, self).matvec(v, c, format=format, axis=axis)

        return c


@inheritdocstrings
class BBDmat(SpectralMatrix):
    """Mass matrix for inner product B_{kj} = (phi_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-4

    and phi_j is the Shen Dirichlet basis function and psi_k the Shen
    Biharmonic basis function.

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

        elif format == 'cython' and v.ndim == 3:
            BBD_matvec3D(v, c, self[-2], self[0], self[2], self[4], axis)

        else:
            c = super(SpectralMatrix, self).matvec(v, c, format=format, axis=axis)

        return c


@inheritdocstrings
class CDNmat(SpectralMatrix):
    """Matrix for inner product C_{kj} = (psi'_j, phi_k)_w

    where

        j = 1, 2, ..., N-2 and k = 0, 1, ..., N-2

    and phi_k is the Shen Dirichlet basis function and psi_j the Shen Neumann
    basis function.

    For simplicity, the matrix is stored including the zero index row (k=0)

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], SD)
        assert isinstance(trial[0], SN)
        N = test[0].N
        ck = get_ck(N, test[0].quad)
        k = np.arange(N-2, dtype=np.float)
        d = {-1: -((k[1:]-1)/(k[1:]+1))**2*(k[1:]+1)*np.pi,
              1: (k[:-1]+1)*np.pi}
        SpectralMatrix.__init__(self, d, test, trial)

    def matvec(self, v, c, format='cython', axis=0):
        if format == 'cython' and v.ndim == 3:
            CDNmat_matvec(self[1], self[-1], v, c, axis)
        else:
            c = super(SpectralMatrix, self).matvec(v, c, format=format, axis=axis)

        return c


@inheritdocstrings
class CDDmat(SpectralMatrix):
    """Matrix for inner product C_{kj} = (phi'_j, phi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-2

    and phi_k is the Shen Dirichlet basis function.

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

        elif format == 'cython' and v.ndim == 3:
            CDDmat_matvec(self[1], self[-1], v, c, axis)
        else:
            c = super(SpectralMatrix, self).matvec(v, c, format=format, axis=axis)

        return c


@inheritdocstrings
class CNDmat(SpectralMatrix):
    """Matrix for inner product C_{kj} = (phi'_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 1, 2, ..., N-2

    and phi_j is the Shen Dirichlet basis function and psi_k the Shen Neumann
    basis function.

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
        c = super(SpectralMatrix, self).matvec(v, c, format=format, axis=axis)
        s = [slice(None),]*v.ndim
        s[axis] = 0
        c[s] = 0
        return c


@inheritdocstrings
class CTDmat(SpectralMatrix):
    """Matrix for inner product C_{kj} = (phi'_j, T_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N

    phi_j is the Shen Dirichlet basis function and T_k is the Chebyshev basis
    function.

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
    """Matrix for inner product C_{kj} = (T'_j, phi_k)_w

    where

        j = 0, 1, ..., N and k = 0, 1, ..., N-2

    phi_k is the Shen Dirichlet basis function and T_j is the Chebyshev basis
    function.

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
    """Matrix for inner product C_{kj} = (phi'_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-4

    phi_j is the Shen Dirichlet basis and psi_k the Shen Biharmonic basis
    function.

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
        elif format == 'cython' and v.ndim == 3:
            CBD_matvec3D(v, c, self[-1], self[1], self[3], axis)
        elif format == 'cython' and v.ndim == 1:
            CBD_matvec(v, c, self[-1], self[1], self[3])
        else:
            c = super(SpectralMatrix, self).matvec(v, c, format=format, axis=axis)
        return c


@inheritdocstrings
class CDBmat(SpectralMatrix):
    """Matrix for inner product C_{kj} = (psi'_j, phi_k)_w

    where

        j = 0, 1, ..., N-4 and k = 0, 1, ..., N-2

    phi_k is the Shen Dirichlet basis function and psi_j the Shen Biharmonic
    basis function.

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

        elif format == 'cython' and v.ndim == 3:
            CDB_matvec3D(v, c, self[-3], self[-1], self[1], axis)

        else:
            c = super(SpectralMatrix, self).matvec(v, c, format=format, axis=axis)

        return c


@inheritdocstrings
class ABBmat(SpectralMatrix):
    """Stiffness matrix for inner product A_{kj} = (psi''_j, psi_k)_w

    where

        j = 0, 1, ..., N-4 and k = 0, 1, ..., N-4

    and psi_k is the Shen Biharmonic basis function.

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

        elif format == 'cython' and v.ndim == 3:
            Tridiagonal_matvec3D(v, c, self[-2], self[0], self[2], axis)

        elif format == 'cython' and v.ndim == 1:
            Tridiagonal_matvec(v, c, self[-2], self[0], self[2])

        else:
            c = super(SpectralMatrix, self).matvec(v, c, format=format, axis=axis)

        return c


@inheritdocstrings
class ADDmat(SpectralMatrix):
    """Stiffness matrix for inner product A_{kj} = (psi''_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-2

    and psi_k is the Shen Dirichlet basis function.

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
        if format == 'cython' and v.ndim == 1:
            ADDmat_matvec(v, c, self[0])
        else:
            c = super(SpectralMatrix, self).matvec(v, c, format=format, axis=axis)

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

        self.testfunction[0].bc.apply_after(u, True)
        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if not u is b:
                b = np.moveaxis(b, 0, axis)
        u /= self.scale
        return u


@inheritdocstrings
class ANNmat(SpectralMatrix):
    """Stiffness matrix for inner product A_{kj} = (phi''_j, phi_k)_w

    where

        j = 1, 2, ..., N-2 and k = 1, 2, ..., N-2

    and phi_k is the Shen Neumann basis function.

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
        c = super(SpectralMatrix, self).matvec(v, c, format=format, axis=axis)
        s = [slice(None),]*v.ndim
        s[axis] = 0
        c[s] = self.testfunction[0].mean*np.pi
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
            b = np.moveaxis(b, axis, 0)
            u = np.moveaxis(u, axis, 0)

        bs = b[s]
        us = u[s]
        j2 = np.arange(self.shape[0])**2
        j2[0] = 1
        j2 = 1./j2
        d = self[0]*j2
        d1 = self[2]*j2[2:]
        if len(b.shape) == 1:
            se = 0.0
            so = 0.0
        else:
            se = np.zeros(u.shape[1:])
            so = np.zeros(u.shape[1:])
            j2.repeat(np.prod(bs.shape[1:])).reshape(bs.shape)

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
        us *= j2

        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if not u is b:
                b = np.moveaxis(b, 0, axis)
        u /= self.scale
        return u


@inheritdocstrings
class ATTmat(SpectralMatrix):
    """Stiffness matrix for inner product A_{kj} = (psi''_j, psi_k)_w

    where

        j = 0, 1, ..., N and k = 0, 1, ..., N

    and psi_k is the Chebyshev basis function.

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
    """Biharmonic matrix for inner product S_{kj} = (psi''''_j, psi_k)_w

    where

        j = 0, 1, ..., N-4 and k = 0, 1, ..., N-4

    and psi_k is the Shen Biharmonic basis function.

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
            SBBmat_matvec3D(v, c, self[0], axis)

        elif format == 'cython' and v.ndim == 1:
            SBBmat_matvec(v, c, self[0])

        else:
            c = super(SpectralMatrix, self).matvec(v, c, format=format, axis=axis)

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
