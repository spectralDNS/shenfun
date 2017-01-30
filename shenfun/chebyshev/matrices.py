from __future__ import division
import numpy as np
from shenfun.optimization.Matvec import CDNmat_matvec, BDNmat_matvec, CDDmat_matvec, \
    SBBmat_matvec, SBBmat_matvec3D, Tridiagonal_matvec, \
    Tridiagonal_matvec3D, Pentadiagonal_matvec, Pentadiagonal_matvec3D, \
    CBD_matvec3D, CBD_matvec, CDB_matvec3D, ADDmat_matvec, \
    BBD_matvec3D

from shenfun.matrixbase import ShenMatrix
from shenfun import inheritdocstrings
from shenfun.la import TDMA, PDMA
from . import bases


@inheritdocstrings
class BDDmat(ShenMatrix):
    """Matrix for inner product B_{kj}=(phi_j, phi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-2

    and phi_j is a Shen Dirichlet basis function.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        d = {0: np.pi/2*(ck[:-2]+ck[2:]),
             2: np.array([-np.pi/2])}
        d[-2] = d[2]
        trial = bases.ShenDirichletBasis(quad=quad)
        ShenMatrix.__init__(self, d, N, (trial, 0), (trial, 0))
        self.solve = TDMA(self)

    def matvec(self, v, c, format='cython'):
        N, M = self.shape
        c.fill(0)
        if format == 'cython':
            ld = self[-2]*np.ones(M-2)
            if v.ndim == 3:
                Tridiagonal_matvec3D(v, c, ld, self[0], ld)
            elif v.ndim == 1:
                Tridiagonal_matvec(v, c, ld, self[0], ld)
            else:
                c = super(BDDmat, self).matvec(v, c, format=format)
                #raise NotImplementedError

        elif format == 'self':
            s = (slice(None),)+(np.newaxis,)*(v.ndim-1) # broadcasting
            c[:(N-2)] = self[2]*v[2:N]
            c[:N] += self[0][s]*v[:N]
            c[2:N] += self[-2]*v[:(N-2)]

        else:
            c = super(BDDmat, self).matvec(v, c, format=format)

        return c


@inheritdocstrings
class BNDmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (phi_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 1, 2, ..., N-2

    psi_k is the Shen Dirichlet basis function and phi_j is a Shen Neumann
    basis function.

    For simplicity, the matrix is stored including the zero index row (k=0)

    """

    def __init__(self, K, quad='GC'):
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        k = K[:N-2].astype(np.float)
        d = {-2: -np.pi/2,
              0: np.pi/2.*(ck[:-2]+ck[2:]*(k/(k+2))**2),
              2: -np.pi/2*(k[:N-4]/(k[:N-4]+2))**2}
        trial = bases.ShenDirichletBasis(quad=quad)
        test = bases.ShenNeumannBasis(quad=quad)
        ShenMatrix.__init__(self, d, N, (trial, 0), (test, 0))

    def matvec(self, v, c, format='csr'):
        c = ShenMatrix.matvec(self, v, c, format=format)
        c[0] = 0
        return c


@inheritdocstrings
class BDNmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (psi_j, phi_k)_w

    where

        j = 1, 2, ..., N-2 and k = 0, 1, ..., N-2

    psi_j is the Shen Dirichlet basis function and phi_k is a Shen Neumann
    basis function.

    For simplicity, the matrix is stored including the zero index column (j=0)

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        k = K[:N-2].astype(np.float)
        d = {-2: -np.pi/2*(k[:N-4]/(k[:N-4]+2))**2,
              0:  np.pi/2.*(ck[:-2]+ck[2:]*(k/(k+2))**2),
              2: -np.pi/2}
        trial = bases.ShenNeumannBasis(quad=quad)
        test = bases.ShenDirichletBasis(quad=quad)
        ShenMatrix.__init__(self, d, N, (trial, 0), (test, 0))

    def matvec(self, v, c, format='cython'):
        c.fill(0)
        if format == 'cython' and v.ndim == 3:
            BDNmat_matvec(self[2], self[-2], self[0], v, c)

        else:
            c = super(BDNmat, self).matvec(v, c, format=format)

        return c


@inheritdocstrings
class BTTmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (T_j, T_k)_w

    where

        j = 0, 1, ..., N and k = 0, 1, ..., N

    and T_j is the jth order Chebyshev function of the first kind.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        trial = bases.ChebyshevBasis(quad)
        ShenMatrix.__init__(self, {0: np.pi/2*ck}, N, (trial, 0), (trial, 0))

    def matvec(self, v, c, format='self'):
        c.fill(0)
        if format == 'self':
            s = (slice(None),)+(np.newaxis,)*(v.ndim-1) # broadcasting
            c[:] = self[0][s]*v
        else:
            c = ShenMatrix.matvec(self, v, c, format=format)

        return c


@inheritdocstrings
class BNNmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (phi_j, phi_k)_w

    where

        j = 1, 2, ..., N-2 and k = 1, 2, ..., N-2

    and phi_j is the Shen Neumann basis function.

    The matrix is stored including the zero index row and column

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        k = K[:-2].astype(np.float)
        d = {0: np.pi/2*(ck[:-2]+ck[2:]*(k[:]/(k[:]+2))**4),
             2: -np.pi/2*((k[2:]-2)/(k[2:]))**2}
        d[-2] = d[2]
        trial = bases.ShenNeumannBasis(quad)
        ShenMatrix.__init__(self, d, N, (trial, 0), (trial, 0))
        self.solve = TDMA(self)

    def matvec(self, v, c, format='csr'):
        c = ShenMatrix.matvec(self, v, c, format=format)
        c[0] = 0
        return c


class BDTmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (T_j, phi_k)_w

    where

        j = 0, 1, ..., N and k = 0, 1, ..., N-2

    phi_k is the Shen Dirichlet basis function and T_j is the Chebyshev basis
    function.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        d = {0: np.pi/2*ck[:N-2],
             2: -np.pi/2*ck[2:]}
        trial = bases.ChebyshevBasis(quad)
        test = bases.ShenDirichletBasis(quad)
        ShenMatrix.__init__(self, d, N, (trial, 0), (test, 0))


class BTDmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (phi_j, T_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N

    phi_j is the Shen Dirichlet basis function and T_k is the Chebyshev basis
    function.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        d = {-2: -np.pi/2*ck[2:],
              0: np.pi/2*ck[:N-2]}
        test = bases.ChebyshevBasis(quad)
        trial = bases.ShenDirichletBasis(quad)
        ShenMatrix.__init__(self, d, N, (trial, 0), (test, 0))


class BTNmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (phi_j, T_k)_w

    where

        j = 1, 2, ..., N-2 and k = 0, 1, ..., N

    phi_J is the Shen Neumann basis function and T_k is the Chebyshev basis
    function.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        d = {-2: -np.pi/2*ck[2:]*((K[2:]-2)/K[2:])**2,
              0: np.pi/2*ck[:-2]}
        trial = bases.ShenNeumannBasis(quad)
        test = bases.ChebyshevBasis(quad)
        ShenMatrix.__init__(self, d, N, (trial, 0), (test, 0))


@inheritdocstrings
class BBBmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (psi_j, psi_k)_w

    where

        j = 0, 1, ..., N-4 and k = 0, 1, ..., N-4

    and psi_j is the Shen Biharmonic basis function.

    """

    def __init__(self, K, quad='GC'):
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        k = K[:N-4].astype(np.float)

        d = {4: (k[:-4]+1)/(k[:-4]+3)*np.pi/2,
             2: -((k[:-2]+2)/(k[:-2]+3) + (k[:-2]+4)*(k[:-2]+1)/((k[:-2]+5)*(k[:-2]+3)))*np.pi,
             0: (ck[:N-4] + 4*((k+2)/(k+3))**2 + ck[4:]*((k+1)/(k+3))**2)*np.pi/2.}
        d[-2] = d[2]
        d[-4] = d[4]
        trial = bases.ShenBiharmonicBasis(quad)
        ShenMatrix.__init__(self, d, N, (trial, 0), (trial, 0))
        self.solve = PDMA(self)

    def matvec(self, v, c, format='cython'):
        c.fill(0)
        N = self.shape[0]
        if format == 'self':
            vv = v[:-4]
            s = (slice(None),) + (np.newaxis,)*(v.ndim-1) # broadcasting
            c[:N] = self[0][s] * vv[:]
            c[:N-2] += self[2][s] * vv[2:]
            c[:N-4] += self[4][s] * vv[4:]
            c[2:N] += self[-2][s] * vv[:-2]
            c[4:N] += self[-4][s] * vv[:-4]

        elif format == 'cython':
            if v.ndim == 3:
                Pentadiagonal_matvec3D(v, c, self[-4], self[-2], self[0],
                                       self[2], self[4])
            elif v.ndim == 1:
                Pentadiagonal_matvec(v, c, self[-4], self[-2], self[0],
                                     self[2], self[4])
            else:
                c = ShenMatrix.matvec(self, v, c, format=format)

        else:
            c = ShenMatrix.matvec(self, v, c, format=format)

        return c


@inheritdocstrings
class BBDmat(ShenMatrix):
    """Mass matrix for inner product B_{kj} = (phi_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-4

    and phi_j is the Shen Dirichlet basis function and psi_k the Shen
    Biharmonic basis function.

    """


    def __init__(self, K, quad='GC'):
        N = K.shape[0]
        ck = self.get_ck(N, quad)
        k = K[:N-4].astype(np.float)
        a = 2*(k+2)/(k+3)
        b = (k[:N-4]+1)/(k[:N-4]+3)
        d = {-2: -np.pi/2,
              0: (ck[:N-4] + a)*np.pi/2,
              2: -(a+b*ck[4:])*np.pi/2,
              4: b[:-2]*np.pi/2}
        trial = bases.ShenDirichletBasis(quad)
        test = bases.ShenBiharmonicBasis(quad)
        ShenMatrix.__init__(self, d, N, (trial, 0), (test, 0))

    def matvec(self, v, c, format='cython'):
        c.fill(0)
        N = self.shape[0]
        if format == 'self':
            vv = v[:-2]
            s = (slice(None),) + (np.newaxis,)*(v.ndim-1) # broadcasting
            c[:N] = self[0][s] * vv[:-2]
            c[:N] += self[2][s] * vv[2:]
            c[:N-2] += self[4][s] * vv[4:]
            c[2:N] += self[-2] * vv[:-4]

        elif format == 'cython' and v.ndim == 3:
            BBD_matvec3D(v, c, self[-2], self[0], self[2], self[4])

        else:
            c = ShenMatrix.matvec(self, v, c, format=format)

        return c


@inheritdocstrings
class CDNmat(ShenMatrix):
    """Matrix for inner product C_{kj} = (psi'_j, phi_k)_w

    where

        j = 1, 2, ..., N-2 and k = 0, 1, ..., N-2

    and phi_k is the Shen Dirichlet basis function and psi_j the Shen Neumann
    basis function.

    For simplicity, the matrix is stored including the zero index row (k=0)

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        k = K[:N-2].astype(np.float)
        d = {-1: -((k[1:]-1)/(k[1:]+1))**2*(k[1:]+1)*np.pi,
              1: (k[:-1]+1)*np.pi}
        trial = bases.ShenNeumannBasis()
        test = bases.ShenDirichletBasis()
        ShenMatrix.__init__(self, d, N, (trial, 1), (test, 0))

    def matvec(self, v, c, format='cython'):
        if format == 'cython' and v.ndim == 3:
            CDNmat_matvec(self[1], self[-1], v, c)
        else:
            c = ShenMatrix.matvec(self, v, c, format=format)

        return c


@inheritdocstrings
class CDDmat(ShenMatrix):
    """Matrix for inner product C_{kj} = (phi'_j, phi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-2

    and phi_k is the Shen Dirichlet basis function.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        d = {-1: -(K[1:N-2]+1)*np.pi,
              1: (K[:(N-3)]+1)*np.pi}
        trial = bases.ShenDirichletBasis()
        ShenMatrix.__init__(self, d, N, (trial, 1), (trial, 0))

    def matvec(self, v, c, format='cython'):
        N = self.shape[0]
        c.fill(0)
        if format == 'self':
            s = (slice(None),) + (np.newaxis,)*(v.ndim-1) # broadcasting
            c[:N-1] = self[1][s]*v[1:N]
            c[1:N] += self[-1][s]*v[:(N-1)]
        elif format == 'cython' and v.ndim == 3:
            CDDmat_matvec(self[1], self[-1], v, c)
        else:
            c = ShenMatrix.matvec(self, v, c, format=format)

        return c


@inheritdocstrings
class CNDmat(ShenMatrix):
    """Matrix for inner product C_{kj} = (phi'_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 1, 2, ..., N-2

    and phi_j is the Shen Dirichlet basis function and psi_k the Shen Neumann
    basis function.

    For simplicity, the matrix is stored including the zero index coloumn (j=0)

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        k = K[:N-2].astype(np.float)
        d = {-1: -(k[1:]+1)*np.pi,
              1: -(2-k[:-1]**2/(k[:-1]+2)**2*(k[:-1]+3))*np.pi}
        for i in range(3, N-1, 2):
            d[i] = -(1-k[:-i]**2/(k[:-i]+2)**2)*2*np.pi
        trial = bases.ShenDirichletBasis()
        test = bases.ShenNeumannBasis()
        ShenMatrix.__init__(self, d, N, (trial, 1), (test, 0))

    def matvec(self, v, c, format='csr'):
        c = ShenMatrix.matvec(self, v, c, format=format)
        c[0] = 0
        return c


class CTDmat(ShenMatrix):
    """Matrix for inner product C_{kj} = (phi'_j, T_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N

    phi_j is the Shen Dirichlet basis function and T_k is the Chebyshev basis
    function.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        d = {-1: -(K[1:N-1]+1)*np.pi,
              1: -2*np.pi}
        for i in range(3, N-2, 2):
            d[i] = -2*np.pi
        trial = bases.ShenDirichletBasis()
        test = bases.ChebyshevBasis()
        ShenMatrix.__init__(self, d, N, (trial, 1), (test, 0))


class CDTmat(ShenMatrix):
    """Matrix for inner product C_{kj} = (T'_j, phi_k)_w

    where

        j = 0, 1, ..., N and k = 0, 1, ..., N-2

    phi_k is the Shen Dirichlet basis function and T_j is the Chebyshev basis
    function.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        d = {1: np.pi*(K[:N-2]+1)}
        trial = bases.ChebyshevBasis()
        test = bases.ShenDirichletBasis()
        ShenMatrix.__init__(self, d, N, (trial, 1), (test, 0))


@inheritdocstrings
class CBDmat(ShenMatrix):
    """Matrix for inner product C_{kj} = (phi'_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-4

    phi_j is the Shen Dirichlet basis and psi_k the Shen Biharmonic basis
    function.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        d = {-1: -(K[1:N-4]+1)*np.pi,
              1: 2*(K[:N-4]+1)*np.pi,
              3: -(K[:N-5]+1)*np.pi}
        trial = bases.ShenDirichletBasis()
        test = bases.ShenBiharmonicBasis()
        ShenMatrix.__init__(self, d, N, (trial, 1), (test, 0))

    def matvec(self, v, c, format='cython'):
        N, M = self.shape
        c.fill(0)
        if format == 'self':
            s = (slice(None),) + (np.newaxis,)*(v.ndim-1) # broadcasting
            c[1:N] = self[-1][s]*v[:M-3]
            c[:N] += self[1][s]*v[1:M-1]
            c[:N-1] += self[3][s]*v[3:M]
        elif format == 'cython' and v.ndim == 3:
            CBD_matvec3D(v, c, self[-1], self[1], self[3])
        elif format == 'cython' and v.ndim == 1:
            CBD_matvec(v, c, self[-1], self[1], self[3])
        else:
            c = ShenMatrix.matvec(self, v, c, format=format)
        return c


@inheritdocstrings
class CDBmat(ShenMatrix):
    """Matrix for inner product C_{kj} = (psi'_j, phi_k)_w

    where

        j = 0, 1, ..., N-4 and k = 0, 1, ..., N-2

    phi_k is the Shen Dirichlet basis function and psi_j the Shen Biharmonic
    basis function.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        d = {-3: (K[3:-2]-2)*(K[3:-2]+1)/K[3:-2]*np.pi,
             -1: -2*(K[1:-3]+1)**2/(K[1:-3]+2)*np.pi,
              1: (K[:-5]+1)*np.pi}
        trial = bases.ShenBiharmonicBasis()
        test = bases.ShenDirichletBasis()
        ShenMatrix.__init__(self, d, N, (trial, 1), (test, 0))

    def matvec(self, v, c, format='cython'):
        N, M = self.shape
        c.fill(0)
        if format == 'self':
            s = (slice(None),) + (np.newaxis,)*(v.ndim-1) # broadcasting
            c[3:N] = self[-3][s] * v[:M-1]
            c[1:N-1] += self[-1][s] * v[:M]
            c[:N-3] += self[1][s] * v[1:M]
        elif format == 'cython' and v.ndim == 3:
            CDB_matvec3D(v, c, self[-3], self[-1], self[1])

        else:
            c = ShenMatrix.matvec(self, v, c, format=format)

        return c


@inheritdocstrings
class ABBmat(ShenMatrix):
    """Stiffness matrix for inner product A_{kj} = -(psi''_j, psi_k)_w

    where

        j = 0, 1, ..., N-4 and k = 0, 1, ..., N-4

    and psi_k is the Shen Biharmonic basis function.

    """

    def __init__(self, K, quad='GC'):
        N = K.shape[0]
        ki = K[:N-4]
        k = K[:N-4].astype(np.float)
        d = {-2: 2*(ki[2:]-1)*(ki[2:]+2)*np.pi,
              0: -4*((ki+1)*(ki+2)**2)/(k+3)*np.pi,
              2: 2*(ki[:-2]+1)*(ki[:-2]+2)*np.pi}
        trial = bases.ShenBiharmonicBasis()
        ShenMatrix.__init__(self, d, N, (trial, 2), (trial, 0))

    def matvec(self, v, c, format='cython'):
        N = self.shape[0]
        c.fill(0)
        if format == 'self':
            s = (slice(None),) + (np.newaxis,)*(v.ndim-1) # broadcasting
            c[:N] = self[0][s] * v[:N]
            c[:N-2] += self[2][s] * v[2:N]
            c[2:N] += self[-2][s] * v[:N-2]
        elif format == 'cython' and v.ndim == 3:
            Tridiagonal_matvec3D(v, c, self[-2], self[0], self[2])

        elif format == 'cython' and v.ndim == 1:
            Tridiagonal_matvec(v, c, self[-2], self[0], self[2])

        else:
            c = ShenMatrix.matvec(self, v, c, format=format)

        return c


@inheritdocstrings
class ADDmat(ShenMatrix):
    """Stiffness matrix for inner product A_{kj} = -(psi''_j, psi_k)_w

    where

        j = 0, 1, ..., N-2 and k = 0, 1, ..., N-2

    and psi_k is the Shen Dirichlet basis function.

    """

    def __init__(self, K, quad='GC', scale=-1.):
        assert len(K.shape) == 1
        N = K.shape[0]
        d = {0: -2*np.pi*(K[:N-2]+1)*(K[:N-2]+2)}
        for i in range(2, N-2, 2):
            d[i] = -4*np.pi*(K[:-(i+2)]+1)
        trial = bases.ShenDirichletBasis()
        ShenMatrix.__init__(self, d, N, (trial, 2), (trial, 0), scale)

        # Following storage more efficient, but requires effort in iadd/isub...
        #d = {0: -2*np.pi*(K[:N-2]+1)*(K[:N-2]+2),
             #2: -4*np.pi*(K[:-4]+1)}
        #for i in range(4, N-2, 2):
            #d[i] = d[2][:2-i]

    def matvec(self, v, c, format='cython'):
        c.fill(0)
        if format == 'cython' and v.ndim == 1:
            ADDmat_matvec(v, c, self[0])
        else:
            c = ShenMatrix.matvec(self, v, c, format=format)

        return c

    def solve(self, b, u=None):
        N = self.shape[0] + 2
        assert N == b.shape[0]
        s = self.trialfunction.slice(N)
        bs = b[s]
        if u is None:
            us = np.zeros_like(b[s])
        else:
            assert u.shape == b.shape
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

        if u is None:
            b[s] = us
            return b
        else:
            return u


@inheritdocstrings
class ANNmat(ShenMatrix):
    """Stiffness matrix for inner product A_{kj} = -(phi''_j, phi_k)_w

    where

        j = 1, 2, ..., N-2 and k = 1, 2, ..., N-2

    and phi_k is the Shen Neumann basis function.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        k = K[:-2].astype(np.float)
        d = {0: -2*np.pi*k**2*(k+1)/(k+2)}
        for i in range(2, N-2, 2):
            d[i] = -4*np.pi*(k[:-i]+i)**2*(k[:-i]+1)/(k[:-i]+2)**2
        trial = bases.ShenNeumannBasis()
        ShenMatrix.__init__(self, d, N, (trial, 2), (trial, 0), -1.0)

    def matvec(self, v, c, format='csr'):
        c = ShenMatrix.matvec(self, v, c, format=format)
        c[0] = self.testfunction.mean*np.pi
        return c

    def solve(self, b, u=None):
        N = self.shape[0] + 2
        assert N == b.shape[0]
        s = self.trialfunction.slice(N)
        bs = b[s]
        if u is None:
            us = np.zeros_like(b[s])
        else:
            assert u.shape == b.shape
            us = u[s]

        j2 = np.arange(N-2)**2
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
        us[0] = self.testfunction.mean
        us *= j2
        if u is None:
            b[s] = us
            return b
        else:
            return u


@inheritdocstrings
class ATTmat(ShenMatrix):
    """Stiffness matrix for inner product A_{kj} = -(psi''_j, psi_k)_w

    where

        j = 0, 1, ..., N and k = 0, 1, ..., N

    and psi_k is the Chebyshev basis function.

    """

    def __init__(self, K, quad='GC'):
        assert len(K.shape) == 1
        N = K.shape[0]
        d = {}
        for j in range(2, N, 2):
            d[j] = K[j:]*(K[j:]**2-K[:-j]**2)*np.pi/2.
        trial = bases.ChebyshevBasis()
        ShenMatrix.__init__(self, d, N, (trial, 2), (trial, 0), -1.0)


@inheritdocstrings
class SBBmat(ShenMatrix):
    """Biharmonic matrix for inner product S_{kj} = -(psi''''_j, psi_k)_w

    where

        j = 0, 1, ..., N-4 and k = 0, 1, ..., N-4

    and psi_k is the Shen Biharmonic basis function.

    """

    def __init__(self, K, quad='GC'):
        N = K.shape[0]
        k = K[:N-4].astype(np.float)
        ki = K[:N-4]
        i = 8*(ki+1)**2*(ki+2)*(ki+4)
        d = {0: i * np.pi}
        for j in range(2, N-4, 2):
            i = 8*(ki[:-j]+1)*(ki[:-j]+2)*(ki[:-j]*(ki[:-j]+4)+3*(ki[j:]+2)**2)
            d[j] = np.array(i*np.pi/(k[j:]+3))
        trial = bases.ShenBiharmonicBasis()
        ShenMatrix.__init__(self, d, N, (trial, 4), (trial, 0))

    def matvec(self, v, c, format='cython'):
        c.fill(0)
        if format == 'cython' and v.ndim == 3:
            SBBmat_matvec3D(v, c, self[0])

        elif format == 'cython' and v.ndim == 1:
            SBBmat_matvec(v, c, self[0])

        else:
            c = ShenMatrix.matvec(self, v, c, format=format)

        return c
