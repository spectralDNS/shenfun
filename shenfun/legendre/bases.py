from shenfun.spectralbase import SpectralBase, work
from shenfun.utilities import inheritdocstrings
from numpy.polynomial import legendre as leg
import numpy as np

__all__ = ['LegendreBase', 'LegendreBasis', 'ShenDirichletBasis',
           'ShenBiharmonicBasis', 'ShenNeumannBasis']

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

    def get_mass_matrix(self):
        from .matrices import mat
        return mat[(self, 0), (self, 0)]


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

    def eval(self, x, fk):
        return leg.legval(x, fk)


@inheritdocstrings
class ShenDirichletBasis(LegendreBase):
    """Shen Legendre basis for Dirichlet boundary conditions

    args:
        quad        ('LG',)       Legendre-Gauss
        bc           (a, b)       Boundary conditions at x=(1,-1)

    Transforms are performed along the first dimension of a multidimensional
    array.

    """

    def __init__(self, quad="LG", bc=(0., 0.)):
        LegendreBase.__init__(self, quad)
        self.bc = bc
        self.LT = LegendreBasis(quad)

    def get_vandermonde_basis(self, V):
        P = np.zeros(V.shape)
        P[:, :-2] = V[:, :-2] - V[:, 2:]
        P[:, -2] = (V[:, 0] + V[:, 1])/2
        P[:, -1] = (V[:, 0] - V[:, 1])/2
        return P

    def scalar_product(self, fj, fk, fast_transform=False):
        if fast_transform:
            fk = self.LT.scalar_product(fj, fk)
            c0 = 0.5*(fk[0] + fk[1])
            c1 = 0.5*(fk[0] - fk[1])
            fk[:-2] -= fk[2:]
            fk[-2] = c0
            fk[-1] = c1
        else:
            fk = self.vandermonde_scalar_product(fj, fk)

        return fk

    def forward(self, fj, fk, fast_transform=True):
        fk = self.scalar_product(fj, fk, fast_transform)
        fk[0] -= np.pi/2*(self.bc[0] + self.bc[1])
        fk[1] -= np.pi/4*(self.bc[0] - self.bc[1])
        fk = self.apply_inverse_mass(fk)
        fk[-2] = self.bc[0]
        fk[-1] = self.bc[1]
        return fk

    def evaluate_expansion_all(self, fk, fj):
        w_hat = work[(fk, 0)]
        w_hat[:-2] = fk[:-2]
        w_hat[2:] -= fk[:-2]
        w_hat[0] += 0.5*(self.bc[0] + self.bc[1])
        w_hat[1] += 0.5*(self.bc[0] - self.bc[1])
        fj = self.LT.backward(w_hat, fj)
        return fj

    def slice(self, N):
        return slice(0, N-2)

    def get_shape(self, N):
        return N-2

    def eval(self, x, fk):
        w_hat = work[(fk, 0)]
        f = leg.legval(x, fk[:-2])
        w_hat[2:] = fk[:-2]
        f -= leg.legval(x, w_hat)
        return f + 0.5*(fk[-1]*(1+x)+fk[-2]*(1-x))


@inheritdocstrings
class ShenNeumannBasis(LegendreBase):
    """Shen basis for homogeneous Neumann boundary conditions

    args:
        quad        ('LG')        Legendre-Gauss
        mean           float      Mean value

    Transforms are performed along the first dimension of a multidimensional
    array.

    """

    def __init__(self, quad="LG", mean=0):
        LegendreBase.__init__(self, quad)
        self.mean = mean
        self.LT = LegendreBasis(quad)
        self._factor = np.zeros(0)

    def get_vandermonde_basis(self, V):
        N = V.shape[0]
        P = np.zeros(V.shape)
        k = np.arange(N).astype(np.float)
        P[:, :-2] = V[:, :-2] - (k[:-2]*(k[:-2]+1)/(k[:-2]+2))/(k[:-2]+3)*V[:, 2:]
        return P

    def set_factor_array(self, v):
        if not self._factor.shape == v.shape:
            if len(v.shape) == 3:
                k = self.wavenumbers(v.shape)
            elif len(v.shape) == 1:
                k = self.wavenumbers(v.shape[0])
            self._factor = k*(k+1)/(k+2)/(k+3)

    def scalar_product(self, fj, fk, fast_transform=True):
        if fast_transform:
            fk = self.LT.scalar_product(fj, fk)
            self.set_factor_array(fk)
            fk[:-2] -= self._factor * fk[2:]

        else:
            fk = self.vandermonde_scalar_product(fj, fk)

        fk[0] = self.mean*np.pi
        fk[-2:] = 0
        return fk

    def evaluate_expansion_all(self, fk, fj):
        w_hat = work[(fk, 0)]
        self.set_factor_array(fk)
        w_hat[:-2] = fk[:-2]
        w_hat[2:] -= self._factor*fk[:-2]
        fj = self.LT.backward(w_hat, fj)
        return fj

    def slice(self, N):
        return slice(0, N-2)

    def get_shape(self, N):
        return N-2

    def eval(self, x, fk):
        w_hat = work[(fk, 0)]
        self.set_factor_array(fk)
        f = leg.legval(x, fk[:-2])
        w_hat[2:] = self._factor*fk[:-2]
        f -= leg.legval(x, w_hat)
        return f


@inheritdocstrings
class ShenBiharmonicBasis(LegendreBase):
    """Shen biharmonic basis

    Homogeneous Dirichlet and Neumann boundary conditions.

    args:
        quad        ('LG',)       Legendre-Gauss

    Transforms are performed along the first dimension of a multidimensional
    array.

    """

    def __init__(self, quad="LG"):
        LegendreBase.__init__(self, quad)
        self.LT = LegendreBasis(quad)
        self._factor1 = np.zeros(0)
        self._factor2 = np.zeros(0)

    def get_vandermonde_basis(self, V):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(np.float)[:-4]
        P[:, :-4] = V[:, :-4] - (2*(2*k+5)/(2*k+7))*V[:, 2:-2] + ((2*k+3)/(2*k+7))*V[:, 4:]
        return P

    def set_factor_arrays(self, v):
        if not self._factor1.shape == v[:-4].shape:
            if len(v.shape) > 1:
                k = self.wavenumbers(v.shape)
            elif len(v.shape) == 1:
                k = self.wavenumbers(v.shape[0])

            self._factor1 = (-2*(2*k+5)/(2*k+7)).astype(float)
            self._factor2 = ((2*k+3)/(2*k+7)).astype(float)

    def scalar_product(self, fj, fk, fast_transform=False):
        if fast_transform:
            self.set_factor_arrays(fk)
            Tk = work[(fk, 0)]
            Tk = self.LT.scalar_product(fj, Tk)
            fk[:-4] = Tk[:-4]
            fk[:-4] += self._factor1 * Tk[2:-2]
            fk[:-4] += self._factor2 * Tk[4:]

        else:
            fk = self.vandermonde_scalar_product(fj, fk)

        fk[-4:] = 0
        return fk

    @staticmethod
    #@optimizer
    def set_w_hat(w_hat, fk, f1, f2):
        w_hat[:-4] = fk[:-4]
        w_hat[2:-2] += f1*fk[:-4]
        w_hat[4:] += f2*fk[:-4]
        return w_hat

    def evaluate_expansion_all(self, fk, fj):
        w_hat = work[(fk, 0)]
        self.set_factor_arrays(fk)
        w_hat = ShenBiharmonicBasis.set_w_hat(w_hat, fk, self._factor1, self._factor2)
        fj = self.LT.backward(w_hat, fj)
        return fj

    def slice(self, N):
        return slice(0, N-4)

    def get_shape(self, N):
        return N-4

    def eval(self, x, fk):
        w_hat = work[(fk, 0)]
        self.set_factor_arrays(fk)
        f = leg.legval(x, fk[:-4])
        w_hat[2:-2] = self._factor1*fk[:-4]
        f += leg.legval(x, w_hat[:-2])
        w_hat[4:] = self._factor2*fk[:-4]
        w_hat[:4] = 0
        f += leg.legval(x, w_hat)
        return f
