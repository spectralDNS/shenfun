from shenfun.spectralbase import SpectralBase
from shenfun import inheritdocstrings
from numpy.polynomial import legendre as leg
import numpy as np

@inheritdocstrings
class LegendreBase(SpectralBase):
    """Base class for all Legendre bases

    args:
        quad        ('LG')  Legendre-Gauss

    Transforms are performed along the first dimension of a multidimensional
    array.

    """

    def __init__(self, quad="LG"):
        assert quad in ('LG',)
        SpectralBase.__init__(self, quad)

    def points_and_weights(self, N, quad):
        """Return points and weights of quadrature

        args:
            N      integer      Number of points
            quad ('LG', )       Legendre-Gauss

        """
        if quad == "LG":
            points, weights = leg.leggauss(N)
        else:
            raise NotImplementedError

        return points, weights

    def vandermonde(self, x, N):
        """Return Legendre Vandermonde matrix

        args:
            x               points for evaluation
            N               Number of Legendre polynomials

        """
        return leg.legvander(x, N-1)

    def get_vandermonde_basis_derivative(self, V, k=0):
        """Return k'th derivatives of basis as a Vandermonde matrix

        args:
            V               Legendre Vandermonde matrix

        kwargs:
            k     integer   Use k'th derivative of basis

        """
        N = V.shape[0]
        if k > 0:
            D = np.zeros((N, N))
            D[:-k, :] = leg.legder(np.eye(N), k)
            V = np.dot(V, D)
        return self.get_vandermonde_basis(V)

    def backward(self, fk, fj, fast_transform=False):
        assert fast_transform is False
        fj = SpectralBase.backward(self, fk, fj, False)
        return fj

    def forward(self, fj, fk, fast_transform=False):
        assert fast_transform is False
        fk = SpectralBase.forward(self, fj, fk, False)
        return fk



@inheritdocstrings
class LegendreBasis(LegendreBase):
    """Basis for regular Legendre series

    args:
        quad        ('LG',)       Legendre-Gauss

    Transforms are performed along the first dimension of a multidimensional
    array.

    """

    def __init__(self, quad="LG"):
        LegendreBase.__init__(self, quad)

    def evaluate_expansion_all(self, fk, fj):
        raise NotImplementedError

    def scalar_product(self, fj, fk, fast_transform=False):
        if fast_transform:
            raise NotImplementedError
        else:
            fk = self.vandermonde_scalar_product(fj, fk)

        return fk


@inheritdocstrings
class ShenDirichletBasis(LegendreBase):
    """Shen Legendre basis for Dirichlet boundary conditions

    args:
        quad        ('LG',)       Legendre-Gauss

    Transforms are performed along the first dimension of a multidimensional
    array.

    """

    def __init__(self, quad="LG"):
        LegendreBase.__init__(self, quad)
        self.LT = LegendreBasis(quad)

    def get_vandermonde_basis(self, V):
        P = np.zeros(V.shape)
        P[:, :-2] = V[:, :-2] - V[:, 2:]
        return P

    def scalar_product(self, fj, fk, fast_transform=False):
        if fast_transform:
            fk = self.LT.scalar_product(fj, fk)
            fk[:-2] -= fk[2:]
        else:
            fk = self.vandermonde_scalar_product(fj, fk)

        fk[-2:] = 0     # Last two not used, so set to zero. Even for nonhomogeneous bcs, where they are technically non-zero
        return fk

    def evaluate_expansion_all(self, fk, fj):
        w_hat = work[(fk, 0)]
        w_hat[:-2] = fk[:-2]
        w_hat[2:] -= fk[:-2]
        fj = self.LT.backward(w_hat, fj)
        return fj

    def get_mass_matrix(self):
        from .matrices import BDDmat
        return BDDmat

    def slice(self, N):
        return slice(0, N-2)

    def get_shape(self, N):
        return N-2
