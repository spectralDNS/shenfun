from numpy.polynomial import chebyshev as n_cheb
import numpy as np
from mpiFFT4py import dct
from shenfun.spectralbase import SpectralBase, work
from shenfun.optimization import Cheb
from shenfun.utilities import inheritdocstrings

__all__ = ['ChebyshevBase', 'ChebyshevBasis', 'ShenDirichletBasis',
           'ShenNeumannBasis', 'ShenBiharmonicBasis']

@inheritdocstrings
class ChebyshevBase(SpectralBase):
    """Abstract base class for all Chebyshev bases

    args:
        quad        ('GL', 'GC')  Chebyshev-Gauss-Lobatto or Chebyshev-Gauss

    Transforms are performed along the first dimension of a multidimensional
    array.

    """

    def __init__(self, quad="GC"):
        assert quad in ('GC', 'GL')
        SpectralBase.__init__(self, quad)

    def points_and_weights(self, N, quad):
        """Return points and weights of quadrature

        args:
            N      integer      Number of points
            quad ('GL', 'GC')   Chebyshev-Gauss-Lobatto or Chebyshev-Gauss

        """
        if quad == "GL":
            points = -(n_cheb.chebpts2(N)).astype(float)
            weights = np.zeros(N)+np.pi/(N-1)
            weights[0] /= 2
            weights[-1] /= 2

        elif quad == "GC":
            points, weights = n_cheb.chebgauss(N)
            points = points.astype(float)
            weights = weights.astype(float)

        return points, weights

    def vandermonde(self, x, N):
        """Return Chebyshev Vandermonde matrix

        args:
            x               points for evaluation
            N               Number of Chebyshev polynomials

        """
        return n_cheb.chebvander(x, N-1)

    def get_vandermonde_basis_derivative(self, V, k=0):
        """Return k'th derivative of basis as a Vandermonde matrix

        args:
            V               Chebyshev Vandermonde matrix

        kwargs:
            k    integer    k'th derivative

        """
        N = V.shape[0]
        if k > 0:
            D = np.zeros((N, N))
            D[:-k, :] = n_cheb.chebder(np.eye(N), k)
            V = np.dot(V, D)
        return self.get_vandermonde_basis(V)


@inheritdocstrings
class ChebyshevBasis(ChebyshevBase):
    """Basis for regular Chebyshev series

    args:
        quad        ('GL', 'GC')  Chebyshev-Gauss-Lobatto or Chebyshev-Gauss
        threads          1        Number of threads used by pyfftw
        planner_effort            Planner effort for FFTs.

    Transforms are performed along the first dimension of a multidimensional
    array.

    """

    def __init__(self, quad="GC", threads=1, planner_effort="FFTW_MEASURE"):
        ChebyshevBase.__init__(self, quad)
        self.threads = threads
        self.planner_effort = planner_effort

    def derivative_coefficients(self, fk, ck):
        """Return coefficients of Chebyshev series for c = f'(x)

        args:
            fk            Coefficients of regular Chebyshev series
            ck            Coefficients of derivative of fk-series

        """
        if len(fk.shape) == 1:
            ck = Cheb.derivative_coefficients(fk, ck)
        elif len(fk.shape) == 3:
            ck = Cheb.derivative_coefficients_3D(fk, ck)
        return ck

    def fast_derivative(self, fj, fd):
        """Return derivative of fj = f(x_j) at quadrature points

        args:
            fj   (input)     Function values on quadrature mesh
            fd   (output)    Function derivative on quadrature mesh

        """
        fk = work[(fj, 0)]
        ck = work[(fj, 1)]
        fk = self.forward(fj, fk)
        ck = self.derivative_coefficients(fk, ck)
        fd = self.backward(ck, fd)
        return fd

    def apply_inverse_mass(self, fk):
        """Apply inverse BTT_{kj} = c_k 2/pi \delta_{kj}

        args:
            fk   (input/output)    Expansion coefficients

        """
        if self.quad == 'GC':
            fk *= (2/np.pi)
            fk[0] /= 2

        elif self.quad == 'GL':
            fk *= (2/np.pi)
            fk[0] /= 2
            fk[-1] /= 2

        return fk

    def evaluate_expansion_all(self, fk, fj):
        if self.quad == "GC":
            fj = dct(fk, fj, type=3, axis=0, threads=self.threads,
                     planner_effort=self.planner_effort)
            fj *= 0.5
            fj += fk[0]/2

        elif self.quad == "GL":
            fj = dct(fk, fj, type=1, axis=0, threads=self.threads,
                     planner_effort=self.planner_effort)
            fj *= 0.5
            fj += fk[0]/2
            fj[::2] += fk[-1]/2
            fj[1::2] -= fk[-1]/2

        return fj

    def scalar_product(self, fj, fk, fast_transform=True):
        N = fj.shape[0]
        if fast_transform:
            if self.quad == "GC":
                fk = dct(fj, fk, type=2, axis=0, threads=self.threads,
                         planner_effort=self.planner_effort)
                fk *= (np.pi/(2*N))

            elif self.quad == "GL":
                fk = dct(fj, fk, type=1, axis=0, threads=self.threads,
                         planner_effort=self.planner_effort)
                fk *= (np.pi/(2*(N-1)))
        else:
            fk = self.vandermonde_scalar_product(fj, fk)

        return fk

    def get_mass_matrix(self):
        from .matrices import BTTmat
        return BTTmat

    def eval(self, x, fk):
        return n_cheb.chebval(x, fk)


@inheritdocstrings
class ShenDirichletBasis(ChebyshevBase):
    """Shen basis for Dirichlet boundary conditions

    args:
        quad        ('GL', 'GC')  Chebyshev-Gauss-Lobatto or Chebyshev-Gauss
        threads          1        Number of threads used by pyfftw
        planner_effort            Planner effort for FFTs.
        bc             (a, b)     Boundary conditions at x=(1,-1)

    Transforms are performed along the first dimension of a multidimensional
    array.

    """

    def __init__(self, quad="GC", threads=1, planner_effort="FFTW_MEASURE",
                 bc=(0., 0.)):
        ChebyshevBase.__init__(self, quad)
        self.threads = threads
        self.planner_effort = planner_effort
        self.bc = bc
        self.CT = ChebyshevBasis(quad, threads, planner_effort)

    def get_vandermonde_basis(self, V):
        P = np.zeros(V.shape)
        P[:, :-2] = V[:, :-2] - V[:, 2:]
        P[:, -2] = (V[:, 0] + V[:, 1])/2
        P[:, -1] = (V[:, 0] - V[:, 1])/2
        return P

    def scalar_product(self, fj, fk, fast_transform=True):
        if fast_transform:
            fk = self.CT.scalar_product(fj, fk)
            c0 = 0.5*(fk[0] + fk[1])
            c1 = 0.5*(fk[0] - fk[1])
            fk[:-2] -= fk[2:]
            fk[-2] = c0
            fk[-1] = c1
        else:
            fk = self.vandermonde_scalar_product(fj, fk)

        return fk

    def evaluate_expansion_all(self, fk, fj):
        w_hat = work[(fk, 0)]
        w_hat[:-2] = fk[:-2]
        w_hat[2:] -= fk[:-2]
        w_hat[0] += 0.5*(self.bc[0] + self.bc[1])
        w_hat[1] += 0.5*(self.bc[0] - self.bc[1])
        fj = self.CT.backward(w_hat, fj)
        return fj

    def forward(self, fj, fk, fast_transform=True):
        fk = self.scalar_product(fj, fk, fast_transform)
        fk[0] -= np.pi/2*(self.bc[0] + self.bc[1])
        fk[1] -= np.pi/4*(self.bc[0] - self.bc[1])
        fk = self.apply_inverse_mass(fk)
        fk[-2] = self.bc[0]
        fk[-1] = self.bc[1]
        return fk

    def get_mass_matrix(self):
        from .matrices import BDDmat
        return BDDmat

    def slice(self, N):
        return slice(0, N-2)

    def get_shape(self, N):
        return N-2

    def eval(self, x, fk):
        w_hat = work[(fk, 0)]
        f = n_cheb.chebval(x, fk[:-2])
        w_hat[2:] = fk[:-2]
        f -= n_cheb.chebval(x, w_hat)
        return f + 0.5*(fk[-1]*(1+x)+fk[-2]*(1-x))


@inheritdocstrings
class ShenNeumannBasis(ChebyshevBase):
    """Shen basis for homogeneous Neumann boundary conditions

    args:
        quad        ('GL', 'GC')  Chebyshev-Gauss-Lobatto or Chebyshev-Gauss
        threads          1        Number of threads used by pyfftw
        planner_effort            Planner effort for FFTs.
        mean           float      Mean value

    Transforms are performed along the first dimension of a multidimensional
    array.

    """

    def __init__(self, quad="GC", threads=1, planner_effort="FFTW_MEASURE",
                 mean=0):
        ChebyshevBase.__init__(self, quad)
        self.threads = threads
        self.planner_effort = planner_effort
        self.mean = mean
        self.CT = ChebyshevBasis(quad, threads, planner_effort)
        self._factor = np.zeros(0)

    def get_vandermonde_basis(self, V):
        N = V.shape[0]
        P = np.zeros(V.shape)
        k = np.arange(N).astype(np.float)
        P[:, :-2] = V[:, :-2] - (k[:-2]/(k[:-2]+2))**2*V[:, 2:]
        return P

    def set_factor_array(self, v):
        if not self._factor.shape == v.shape:
            if len(v.shape) == 3:
                k = self.wavenumbers(v.shape)
            elif len(v.shape) == 1:
                k = self.wavenumbers(v.shape[0])
            self._factor = (k/(k+2))**2

    def scalar_product(self, fj, fk, fast_transform=True):
        if fast_transform:
            fk = self.CT.scalar_product(fj, fk)
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
        fj = self.CT.backward(w_hat, fj)
        return fj

    def get_mass_matrix(self):
        from .matrices import BNNmat
        return BNNmat

    def slice(self, N):
        return slice(0, N-2)

    def get_shape(self, N):
        return N-2

    def eval(self, x, fk):
        w_hat = work[(fk, 0)]
        self.set_factor_array(fk)
        f = n_cheb.chebval(x, fk[:-2])
        w_hat[2:] = self._factor*fk[:-2]
        f -= n_cheb.chebval(x, w_hat)
        return f


@inheritdocstrings
class ShenBiharmonicBasis(ChebyshevBase):
    """Shen biharmonic basis

    Homogeneous Dirichlet and Neumann boundary conditions.

    args:
        quad        ('GL', 'GC')  Chebyshev-Gauss-Lobatto or Chebyshev-Gauss
        threads          1        Number of threads used by pyfftw
        planner_effort            Planner effort for FFTs.

    Transforms are performed along the first dimension of a multidimensional
    array.

    """

    def __init__(self, quad="GC", threads=1, planner_effort="FFTW_MEASURE"):
        ChebyshevBase.__init__(self, quad)
        self.threads = threads
        self.planner_effort = planner_effort
        self.CT = ChebyshevBasis(quad, threads, planner_effort)
        self._factor1 = np.zeros(0)
        self._factor2 = np.zeros(0)

    def get_vandermonde_basis(self, V):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(np.float)[:-4]
        P[:, :-4] = V[:, :-4] - (2*(k+2)/(k+3))*V[:, 2:-2] + ((k+1)/(k+3))*V[:, 4:]
        return P

    def set_factor_arrays(self, v):
        if not self._factor1.shape == v[:-4].shape:
            if len(v.shape) > 1:
                k = self.wavenumbers(v.shape)
            elif len(v.shape) == 1:
                k = self.wavenumbers(v.shape[0])

            self._factor1 = (-2*(k+2)/(k+3)).astype(float)
            self._factor2 = ((k+1)/(k+3)).astype(float)

    def scalar_product(self, fj, fk, fast_transform=True):
        if fast_transform:
            self.set_factor_arrays(fk)
            Tk = work[(fk, 0)]
            Tk = self.CT.scalar_product(fj, Tk)
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
        fj = self.CT.backward(w_hat, fj)
        return fj

    def get_mass_matrix(self):
        from .matrices import BBBmat
        return BBBmat

    def slice(self, N):
        return slice(0, N-4)

    def get_shape(self, N):
        return N-4

    def eval(self, x, fk):
        w_hat = work[(fk, 0)]
        self.set_factor_arrays(fk)
        f = n_cheb.chebval(x, fk[:-4])
        w_hat[2:-2] = self._factor1*fk[:-4]
        f += n_cheb.chebval(x, w_hat[:-2])
        w_hat[4:] = self._factor2*fk[:-4]
        w_hat[:4] = 0
        f += n_cheb.chebval(x, w_hat)
        return f
