r"""
This module contains classes for working with the spectral-Galerkin method

There are classes for 6 bases and corresponding function spaces

All bases have expansions

    u(x_j) = \sum_k \hat{u}_k \phi_k

where j = 0, 1, ..., N and k = indexset(basis), and the indexset differs from
base to base, see function space definitions below. \phi_k is the k't basis
function of the basis span(\phi_k, k=indexset(basis)).

Chebyshev:
    ChebyshevBasis:
        basis functions:                 basis:
        \phi_k = T_k                     span(T_k, k=0,1,..., N)

    ShenDirichletBasis:
        basis functions:                 basis:
        \phi_k = T_k-T_{k+2}             span(\phi_k, k=0,1,...,N)
        \phi_{N-1} = 0.5(T_0+T_1)
        \phi_{N} = 0.5(T_0-T_1)

        u(1)=a, u(-1)=b, \hat{u}{N-1}=a, \hat{u}_{N}=b

        Note that there are only N-1 unknown coefficients, \hat{u}_k, since
        \hat{u}_{N-1} and \hat{u}_{N} are determined by boundary conditions.

    ShenNeumannBasis:
        basis function:                  basis:
        \phi_k = T_k-(k/(k+2))**2T_{k+2} span(\phi_k, k=1,2,...,N-2)

        Homogeneous Neumann boundary conditions, u'(\pm 1) = 0, and
        zero weighted mean: \int_{-1}^{1}u(x)w(x)dx = 0

    ShenBiharmonicBasis:
        basis function:
        \phi_k = T_k - (2*(k+2)/(k+3))*T_{k+2} + ((k+1)/(k+3))*T_{k+4}

        basis:
        span(\phi_k, k=0,1,...,N-4)

        Homogeneous Dirichlet and Neumann, u(\pm 1)=0 and u'(\pm 1)=0

Legendre:
    LegendreBasis:
        basis function:                  basis:
        \phi_k = L_k                     span(L_k, k=0,1,...N)

    ShenDirichletBasis:
        basis function:                  basis:
        \phi_k = L_k-L_{k+2}             span(\phi_k, k=0,1,...,N-2)

      Homogeneous Dirichlet boundary conditions, u(\pm 1)=0

Each class has methods for moving fast between spectral and physical space, and
for computing the (weighted) scalar product.

"""
import numpy as np
from .utilities import inheritdocstrings
from mpiFFT4py import work_arrays

work = work_arrays()

class SpectralBase(object):
    """Abstract base class for all spectral function spaces

    args:
        quad   ('GL', 'GC', 'LG')  Chebyshev-Gauss-Lobatto, Chebyshev-Gauss
                                   or Legendre-Gauss

    Transforms are performed along the first dimension of a multidimensional
    array.

    """

    def __init__(self, quad):
        self.quad = quad
        self._mass = np.zeros((0, 0)) # Mass matrix (if needed)

    def points_and_weights(self, N, quad):
        """Return points and weights of quadrature

        args:
            N      integer            Number of points
            quad ('GL', 'GC', 'LG')   Chebyshev-Gauss-Lobatto, Chebyshev-Gauss
                                      or Legendre-Gauss

        """
        raise NotImplementedError

    def wavenumbers(self, N):
        """Return the wavenumbermesh

        The first axis is inhomogeneous, and if ndim(N) > 1, then the trailing
        axes are broadcasted.

        """
        N = list(N) if np.ndim(N) else [N]
        k = np.arange(N[0], dtype=np.float)[self.slice(N[0])]
        return k[(self.slice(N[0]),)+(np.newaxis,)*(len(N)-1)]
        #s = [self.slice(N[0])]
        #for n in N[1:]:
            #s.append(slice(0, n))
        #return np.mgrid.__getitem__(s).astype(float)[0]

    def evaluate_expansion_all(self, fk, fj):
        r"""Evaluate expansion on entire mesh

           f(x_j) = \sum_k f_k \T_k(x_j)  for all j = 0, 1, ..., N

        args:
            fk   (input)     Expansion coefficients
            fj   (output)    Function values on quadrature mesh

        """
        raise NotImplementedError

    def scalar_product(self, fj, fk, fast_transform=True):
        r"""Return scalar product

          f_k = (f, \phi_k)_w      for all k = 0, 1, ..., N
              = \sum_j f(x_j) \phi_k(x_j) \sigma(x_j)

        args:
            fj   (input)     Function values on quadrature mesh
            fk   (output)    Expansion coefficients

        """
        raise NotImplementedError

    def forward(self, fj, fk, fast_transform=True):
        """Fast forward transform

        args:
            fj   (input)     Function values on quadrature mesh
            fk   (output)    Expansion coefficients

        kwargs:
            fast_transform   bool - If True use fast transforms,
                             if False use Vandermonde type

        """
        fk = self.scalar_product(fj, fk, fast_transform)
        fk = self.apply_inverse_mass(fk)
        return fk

    def backward(self, fk, fj, fast_transform=True):
        """Fast backward transform

        args:
            fk   (input)     Expansion coefficients
            fj   (output)    Function values on quadrature mesh

        kwargs:
            fast_transform   bool - If True use fast transforms,
                             if False use Vandermonde type

        """
        if fast_transform:
            fj = self.evaluate_expansion_all(fk, fj)
        else:
            fj = self.vandermonde_evaluate_expansion_all(fk, fj)
        return fj

    def vandermonde(self, x, N):
        """Return Vandermonde matrix

        args:
            x               points for evaluation
            N               Number of polynomials

        """
        raise NotImplementedError

    def get_vandermonde_basis(self, V):
        """Return basis as a Vandermonde matrix

        V is a Vandermonde matrix

        """
        return V

    def get_vandermonde_basis_derivative(self, V, k=0):
        """Return k'th derivative of basis as a Vandermonde matrix

        args:
            V               Vandermonde matrix

        kwargs:
            k    integer    k'th derivative

        """
        raise NotImplementedError

    def vandermonde_scalar_product(self, fj, fk):
        """Naive implementation of scalar product

        args:
            fj   (input)    Function values on quadrature mesh
            fk   (output)   Expansion coefficients

        """
        N = fj.shape[0]
        points, weights = self.points_and_weights(N, self.quad)
        V = self.vandermonde(points, N)
        P = self.get_vandermonde_basis(V)
        if fj.ndim == 1:
            fk[:] = np.dot(fj*weights, P)

        else: # broadcasting
            fc = np.rollaxis(fj*weights[(slice(None),)+(np.newaxis,)*(fj.ndim-1)], 0, fj.ndim)
            fk[:] = np.rollaxis(np.dot(fc, P), fj.ndim-1, 0)

        return fk

    def vandermonde_evaluate_expansion_all(self, fk, fj):
        """Naive implementation of evaluate_expansion_all

        args:
            fk   (input)    Expansion coefficients
            fj   (output)   Function values on quadrature mesh

        """
        N = fj.shape[0]
        points = self.points_and_weights(N, self.quad)[0]
        V = self.vandermonde(points, N)
        P = self.get_vandermonde_basis(V)
        if fj.ndim == 1:
            fj = np.dot(P, fk, out=fj)
        else:
            fc = np.rollaxis(fk, 0, fj.ndim-1)
            fj = np.dot(P, fc, out=fj)

        return fj

    def apply_inverse_mass(self, fk):
        """Apply inverse mass matrix

        args:
            fk   (input/output)    Expansion coefficients. fk is overwritten
                                   by applying the inverse mass matrix, and
                                   returned.

        """
        if self._mass.shape[0] != fk.shape[0]:
            B = self.get_mass_matrix()
            self._mass = B(np.arange(fk.shape[0]).astype(np.float))
        if self._mass.testfunction[0].quad != self.quad:
            B = self.get_mass_matrix()
            self._mass = B(np.arange(fk.shape[0]).astype(np.float))
        fk = self._mass.solve(fk)
        return fk

    def eval(self, x, fk):
        """Evaluate basis at position x

        args:
            x    float or array of floats
            fk   Array of expansion coefficients

        """
        raise NotImplementedError

    def get_mass_matrix(self):
        """Return mass matrix associated with current basis"""
        raise NotImplementedError

    def slice(self, N):
        """Return index set of current basis, with N points in real space"""
        return slice(0, N)

    def get_shape(self, N):
        """Return the shape of current basis used to build a ShenMatrix"""
        return N

    def __hash__(self):
        return hash(repr(self.__class__))

    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__
