"""
Module for function spaces of generalized Jacobi type

Note the configuration setting

    from shenfun.config import config
    config['bases']['jacobi']['mode']

Setting this to 'mpmath' can make use of extended precision.
The precision can also be set in the configuration.

    from mpmath import mp
    mp.dps = config['jacobi'][precision]

where mp.dps is the number of significant digits.

Note that extended precision is costly, but for some of the
matrices that can be created with the Jacobi bases it is necessary.
Also note the the higher precision is only used for assembling
matrices comuted with :func:`evaluate_basis_derivative_all`.
It has no effect for the matrices that are predifined in the
matrices.py module. Also note that the final matrix will be
in regular double precision. So the higher precision is only used
for the intermediate assembly.

"""

import functools
import numpy as np
import sympy as sp
from scipy.special import eval_jacobi, roots_jacobi #, gamma
from mpi4py_fft import fftw
from shenfun.config import config
from shenfun.spectralbase import SpectralBase, Transform, islicedict, slicedict
from shenfun.matrixbase import SparseMatrix
from .recursions import b, h, matpow, n

try:
    import quadpy
    from mpmath import mp
    mp.dps = config['bases']['jacobi']['precision']
    has_quadpy = True
except:
    has_quadpy = False
    mp = None

mode = config['bases']['jacobi']['mode']
mode = mode if has_quadpy else 'numpy'

xp = sp.Symbol('x', real=True)
m, n, k = sp.symbols('m,n,k', real=True, integer=True)

#pylint: disable=method-hidden,no-else-return,not-callable,abstract-method,no-member,cyclic-import

__all__ = ['JacobiBase', 'Orthogonal', 'Phi1', 'Phi2', 'Phi4',
           'CompactDirichlet', 'ShenDirichlet', 'ShenBiharmonic',
           'ShenOrder6', 'mode', 'has_quadpy', 'mp']


class JacobiBase(SpectralBase):
    """Base class for all Jacobi spaces

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - JG - Jacobi-Gauss
        alpha : number, optional
            Parameter of the Jacobi polynomial
        beta : number, optional
            Parameter of the Jacobi polynomial
        domain : 2-tuple of floats, optional
            The computational domain
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))

    """

    def __init__(self, N, quad="JG", alpha=0, beta=0, domain=(-1., 1.), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        SpectralBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                              padding_factor=padding_factor, dealias_direct=dealias_direct,
                              coordinates=coordinates)
        self.alpha = alpha
        self.beta = beta
        self.forward = functools.partial(self.forward, fast_transform=False)
        self.backward = functools.partial(self.backward, fast_transform=False)
        self.scalar_product = functools.partial(self.scalar_product, fast_transform=False)
        self.plan(int(N*padding_factor), 0, dtype, {})

    @staticmethod
    def family():
        return 'jacobi'

    def reference_domain(self):
        return (-1, 1)

    def get_orthogonal(self):
        return Orthogonal(self.N,
                          quad=self.quad,
                          domain=self.domain,
                          dtype=self.dtype,
                          padding_factor=self.padding_factor,
                          dealias_direct=self.dealias_direct,
                          coordinates=self.coors.coordinates,
                          alpha=0,
                          beta=0)

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)
        assert self.quad == "JG"
        points, weights = roots_jacobi(N, self.alpha, self.beta)
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def mpmath_points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if mode == 'numpy' or not has_quadpy:
            return self.points_and_weights(N=N, map_true_domain=map_true_domain, weighted=weighted, **kw)
        if N is None:
            N = self.shape(False)
        pw = quadpy.c1.gauss_jacobi(N, self.alpha, self.beta, 'mpmath')
        points = pw.points_symbolic
        weights = pw.weights_symbolic
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def jacobi(self, x, alpha, beta, N):
        V = np.zeros((x.shape[0], N))
        if mode == 'numpy':
            for n in range(N):
                V[:, n] = eval_jacobi(n, alpha, beta, x)
        else:
            for n in range(N):
                V[:, n] = sp.lambdify(xp, sp.jacobi(n, alpha, beta, xp), 'mpmath')(x)
        return V

    def derivative_jacobi(self, x, alpha, beta, k=1):
        V = self.jacobi(x, alpha+k, beta+k, self.N)
        if k > 0:
            Vc = np.zeros_like(V)
            for j in range(k, self.N):
                dj = np.prod(np.array([j+alpha+beta+1+i for i in range(k)]))
                #dj = gamma(j+alpha+beta+1+k) / gamma(j+alpha+beta+1)
                Vc[:, j] = (dj/2**k)*V[:, j-k]
            V = Vc
        return V

    def vandermonde(self, x):
        return self.jacobi(x, self.alpha, self.beta, self.shape(False))

    def plan(self, shape, axis, dtype, options):
        if shape in (0, (0,)):
            return

        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[0]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        U = fftw.aligned(shape, dtype=dtype)
        V = fftw.aligned(shape, dtype=dtype)
        U.fill(0)
        V.fill(0)
        self.axis = axis
        if self.padding_factor > 1.+1e-8:
            trunc_array = self._get_truncarray(shape, V.dtype)
            self.scalar_product = Transform(self.scalar_product, None, U, V, trunc_array)
            self.forward = Transform(self.forward, None, U, V, trunc_array)
            self.backward = Transform(self.backward, None, trunc_array, V, U)
        else:
            self.scalar_product = Transform(self.scalar_product, None, U, V, V)
            self.forward = Transform(self.forward, None, U, V, V)
            self.backward = Transform(self.backward, None, V, V, U)

        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)


class Orthogonal(JacobiBase):
    """Function space for regular (orthogonal) Jacobi functions

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - JG - Jacobi-Gauss

        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """

    def __init__(self, N, quad="JG", alpha=-0.5, beta=-0.5, domain=(-1., 1.),
                 dtype=float, padding_factor=1, dealias_direct=False, coordinates=None):
        JacobiBase.__init__(self, N, quad=quad, alpha=alpha, beta=beta, domain=domain, dtype=dtype,
                            padding_factor=padding_factor, dealias_direct=dealias_direct,
                            coordinates=coordinates)

    @property
    def is_orthogonal(self):
        return True

    #def get_orthogonal(self):
    #    return self

    @staticmethod
    def short_name():
        return 'J'

    def sympy_basis(self, i=0, x=xp):
        return sp.jacobi(i, self.alpha, self.beta, x)

    def bnd_values(self, k=0):
        if k == 0:
            return (lambda i: (-1)**i*sp.binomial(i+self.beta, i), lambda i: sp.binomial(i+self.alpha, i))
        elif k == 1:
            gam = lambda i: sp.gamma(self.alpha+self.beta+i+2)/(2*sp.gamma(self.alpha+self.beta+i+1))
            return (lambda i: (-1)**(i-1)*gam(i)*sp.binomial(i+self.beta, i-1), lambda i: gam(i)*sp.binomial(i+self.alpha, i-1))

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if mode == 'numpy':
            output_array = eval_jacobi(i, self.alpha, self.beta, x, out=output_array)
        else:
            f = self.sympy_basis(i, xp)
            output_array[:] = sp.lambdify(xp, f, 'mpmath')(x)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.points_and_weights(mode=mode)[0]
        #x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)

        if mode == 'numpy':
            dj = np.prod(np.array([i+self.alpha+self.beta+1+j for j in range(k)]))
            output_array[:] = dj/2**k*eval_jacobi(i-k, self.alpha+k, self.beta+k, x)
        else:
            f = sp.jacobi(i, self.alpha, self.beta, xp)
            output_array[:] = sp.lambdify(xp, f.diff(xp, k), 'mpmath')(x)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mpmath_points_and_weights(mode=mode)[0]
        if mode == 'numpy':
            return self.derivative_jacobi(x, self.alpha, self.beta, k)
        else:
            N = self.shape(False)
            V = np.zeros((x.shape[0], N))
            for i in range(N):
                f = sp.jacobi(i, self.alpha, self.beta, xp)
                V[:, i] = sp.lambdify(xp, f.diff(xp, k), 'mpmath')(x)
        return V

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            x = self.mpmath_points_and_weights()[0]
        return self.vandermonde(x)

class CompositeSpace(Orthogonal):
    """Common class for all spaces based on composite bases"""

    def __init__(self, N, quad="JG", bc=(0, 0), domain=(-1., 1.), dtype=float, alpha=0, beta=0,
                 scaled=False, padding_factor=1, dealias_direct=False, coordinates=None):
        Orthogonal.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                            padding_factor=padding_factor, dealias_direct=dealias_direct,
                            alpha=alpha, beta=beta, coordinates=coordinates)
        JacobiBase.plan(self, (int(padding_factor*N),), 0, dtype, {})
        from shenfun.tensorproductspace import BoundaryValues
        self._bc_basis = None
        self._scaled = scaled
        self.bc = BoundaryValues(self, bc=bc)

    def plan(self, shape, axis, dtype, options):
        Orthogonal.plan(self, shape, axis, dtype, options)
        JacobiBase.plan(self, shape, axis, dtype, options)

    def evaluate_basis_all(self, x=None, argument=0):
        V = Orthogonal.evaluate_basis_all(self, x=x, argument=argument)
        return self._composite(V, argument=argument)

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        V = Orthogonal.evaluate_basis_derivative_all(self, x=x, k=k, argument=argument)
        return self._composite(V, argument=argument)

    def _evaluate_expansion_all(self, input_array, output_array, x=None, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_expansion_all(self, input_array, output_array, x, False)
            return
        assert input_array is self.backward.tmp_array
        assert output_array is self.backward.output_array
        input_array[:] = self.to_ortho(input_array, output_array)
        Orthogonal._evaluate_expansion_all(self, input_array, output_array, x, fast_transform)

    def _evaluate_scalar_product(self, fast_transform=True):
        output = self.scalar_product.tmp_array
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            output[self.sl[slice(-(self.N-self.dim()), None)]] = 0
            return
        Orthogonal._evaluate_scalar_product(self, True)
        K = self.stencil_matrix(self.shape(False))
        w0 = output.copy()
        output = K.matvec(w0, output, axis=self.axis)

    @property
    def is_orthogonal(self):
        return False

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def is_scaled(self):
        """Return True if scaled basis is used, otherwise False"""
        return self._scaled

    def stencil_matrix(self, N=None):
        """Matrix describing the linear combination of orthogonal basis
        functions for the current basis.

        Parameters
        ----------
        N : int, optional
            The number of quadrature points
        """
        raise NotImplementedError

    def _composite(self, V, argument=0):
        """Return Vandermonde matrix V adjusted for basis composition

        Parameters
        ----------
        V : Vandermonde type matrix
        argument : int
            Zero for test and 1 for trialfunction

        """
        P = np.zeros_like(V)
        P[:] = V * self.stencil_matrix(V.shape[1]).diags().T
        if argument == 1: # if trial function
            P[:, slice(-(self.N-self.dim()), None)] = self.get_bc_basis()._composite(V)
        return P

    def sympy_basis(self, i=0, x=xp):
        assert i < self.N
        if i < self.dim():
            row = self.stencil_matrix().diags().getrow(i)
            f = 0
            for j, val in zip(row.indices, row.data):
                f += sp.nsimplify(val)*Orthogonal.sympy_basis(self, i=j, x=x)
        else:
            f = self.get_bc_basis().sympy_basis(i=i-self.dim(), x=x)
        return f

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)
        s = [np.newaxis]*self.dimensions
        for key, val in self.stencil_matrix().items():
            M = self.N if key >= 0 else self.dim()
            s0 = slice(max(0, -key), min(self.dim(), M-max(0, key)))
            Q = s0.stop-s0.start
            s1 = slice(max(0, key), max(0, key)+Q)
            s[self.axis] = slice(0, Q)
            output_array[self.sl[s1]] += val[tuple(s)]*input_array[self.sl[s0]]
        if self.has_nonhomogeneous_bcs:
            self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i < self.dim():
            row = self.stencil_matrix().diags().getrow(i)
            w0 = np.zeros_like(output_array)
            output_array.fill(0)
            for j, val in zip(row.indices, row.data):
                output_array[:] += val*Orthogonal.evaluate_basis(self, x, i=j, output_array=w0)
        else:
            assert i < self.N
            output_array = self.get_bc_basis().evaluate_basis(x, i=i-self.dim(), output_array=output_array)
        return output_array

    def evaluate_basis_derivative(self, x, i=0, k=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i < self.dim():
            row = self.stencil_matrix().diags().getrow(i)
            w0 = np.zeros_like(output_array)
            output_array.fill(0)
            for j, val in zip(row.indices, row.data):
                output_array[:] += val*Orthogonal.evaluate_basis_derivative(self, x, i=j, k=k, output_array=w0)
        else:
            assert i < self.N
            output_array = self.get_bc_basis().evaluate_basis_derivative(x, i=i-self.dim(), k=k, output_array=output_array)
        return output_array

class Phi1(CompositeSpace):
    def __init__(self, N, quad="JG", bc=(0., 0.), domain=(-1., 1.), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, alpha=0, beta=0, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                alpha=alpha, beta=beta, coordinates=coordinates)
        self.b0n = sp.simplify(b(alpha, beta, n+1, n) / h(alpha, beta, n, 0))
        if not alpha == beta:
            self.b1n = sp.simplify(b(alpha, beta, n+1, n+1) / h(alpha, beta, n+1, 0))
        self.b2n = sp.simplify(b(alpha, beta, n+1, n+2) / h(alpha, beta, n+2, 0))

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'P1'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        k = np.arange(N)
        d0, d2 = np.zeros(N), np.zeros(N-2)
        d0[:-2] = sp.lambdify(n, self.b0n)(k[:N-2])
        d2[:] = sp.lambdify(n, self.b2n)(k[:N-2])
        if not self.alpha == self.beta:
            d1 = np.zeros(N-1)
            d1[:-1] = sp.lambdify(n, self.b1n)(k[:N-2])
            return SparseMatrix({0: d0, 1: d1, 2: d2}, (N, N))
        return SparseMatrix({0: d0, 2: d2}, (N, N))

    def slice(self):
        return slice(0, self.N-2)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCDirichlet(self.N, quad=self.quad, domain=self.domain,
                                     scaled=self._scaled, alpha=self.alpha, beta=self.beta,
                                     coordinates=self.coors.coordinates)
        return self._bc_basis

class Phi2(CompositeSpace):
    def __init__(self, N, quad="JG", bc=(0, 0, 0, 0), domain=(-1., 1.), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, alpha=0, beta=0, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                alpha=alpha, beta=beta, coordinates=coordinates)
        self.b0n = sp.simplify(matpow(b, 2, alpha, beta, n+2, n) / h(alpha, beta, n, 0))
        self.b1n = None
        self.b2n = sp.simplify(matpow(b, 2, alpha, beta, n+2, n+2) / h(alpha, beta, n+2, 0))
        self.b3n = None
        self.b4n = sp.simplify(matpow(b, 2, alpha, beta, n+2, n+4) / h(alpha, beta, n+4, 0))
        if not alpha == beta:
            self.b1n = sp.simplify(matpow(b, 2, alpha, beta, n+2, n+1) / h(alpha, beta, n+1, 0))
            self.b3n = sp.simplify(matpow(b, 2, alpha, beta, n+2, n+3) / h(alpha, beta, n+3, 0))

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    @staticmethod
    def short_name():
        return 'P2'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        k = np.arange(N)
        d0, d2, d4 = np.zeros(N), np.zeros(N-2), np.zeros(N-4)
        d0[:-4] = sp.lambdify(n, self.b0n)(k[:N-4])
        d2[:-2] = sp.lambdify(n, self.b2n)(k[:N-4])
        d4[:] = sp.lambdify(n, self.b4n)(k[:N-4])
        if self.b1n:
            d1, d3 = np.zeros(N-1), np.zeros(N-3)
            d1[:-3] = sp.lambdify(n, self.b1n)(k[:N-4])
            d3[:-1] = sp.lambdify(n, self.b3n)(k[:N-4])
            return SparseMatrix({0: d0, 1: d1, 2: d2, 3: d3, 4: d4}, (N, N))
        return SparseMatrix({0: d0, 2: d2, 4: d4}, (N, N))

    def slice(self):
        return slice(0, self.N-4)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCBiharmonic(self.N, quad=self.quad, domain=self.domain,
                                      scaled=self._scaled, alpha=self.alpha, beta=self.beta,
                                      coordinates=self.coors.coordinates)
        return self._bc_basis

class Phi4(CompositeSpace):
    def __init__(self, N, quad="GC", bc=(0,)*8, domain=(-1, 1), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, alpha=0, beta=0, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                alpha=alpha, beta=beta, coordinates=coordinates)
        self.b0n = sp.simplify(matpow(b, 4, alpha, beta, n+4, n) / h(alpha, beta, n, 0))
        self.b2n = sp.simplify(matpow(b, 4, alpha, beta, n+4, n+2) / h(alpha, beta, n+2, 0))
        self.b4n = sp.simplify(matpow(b, 4, alpha, beta, n+4, n+4) / h(alpha, beta, n+4, 0))
        self.b6n = sp.simplify(matpow(b, 4, alpha, beta, n+4, n+6) / h(alpha, beta, n+6, 0))
        self.b8n = sp.simplify(matpow(b, 4, alpha, beta, n+4, n+8) / h(alpha, beta, n+8, 0))
        self.b1n, self.b3n, self.b5n, self.b7n = (None,)*4
        if not alpha == beta:
            self.b1n = sp.simplify(matpow(b, 4, alpha, beta, n+4, n+1) / h(alpha, beta, n+1, 0))
            self.b3n = sp.simplify(matpow(b, 4, alpha, beta, n+4, n+3) / h(alpha, beta, n+3, 0))
            self.b5n = sp.simplify(matpow(b, 4, alpha, beta, n+4, n+5) / h(alpha, beta, n+5, 0))
            self.b7n = sp.simplify(matpow(b, 4, alpha, beta, n+4, n+7) / h(alpha, beta, n+7, 0))

    @staticmethod
    def boundary_condition():
        return 'Biharmonic*2'

    @staticmethod
    def short_name():
        return 'P4'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        k = np.arange(N)
        d0, d2, d4, d6, d8 = np.zeros(N), np.zeros(N-2), np.zeros(N-4), np.zeros(N-6), np.zeros(N-8)
        d0[:-8] = sp.lambdify(n, self.b0n)(k[:N-8])
        d2[:-6] = sp.lambdify(n, self.b2n)(k[:N-8])
        d4[:-4] = sp.lambdify(n, self.b4n)(k[:N-8])
        d6[:-2] = sp.lambdify(n, self.b6n)(k[:N-8])
        d8[:] = sp.lambdify(n, self.b8n)(k[:N-8])
        if not self.alpha == self.beta:
            d1, d3, d5, d7 = np.zeros(N-1), np.zeros(N-3), np.zeros(N-5), np.zeros(N-7)
            d1[:-7] = sp.lambdify(n, self.b1n)(k[:N-8])
            d3[:-5] = sp.lambdify(n, self.b3n)(k[:N-8])
            d5[:-3] = sp.lambdify(n, self.b5n)(k[:N-8])
            d7[:-1] = sp.lambdify(n, self.b7n)(k[:N-8])
            return SparseMatrix({0: d0, 1: d2, 2: d2, 3: d3, 4: d4, 5: d5, 6: d6, 7: d7, 8: d8}, (N, N))
        return SparseMatrix({0: d0, 2: d2, 4: d4, 6: d6, 8: d8}, (N, N))

    def slice(self):
        return slice(0, self.N-8)

    def get_bc_basis(self):
        raise NotImplementedError
        # This basis should probably only be used as test function, and thus no inhomogeneous bcs required


class CompactDirichlet(CompositeSpace):
    def __init__(self, N, quad="JG", bc=(0., 0.), domain=(-1., 1.), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, alpha=0, beta=0, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                alpha=alpha, beta=beta, coordinates=coordinates)
        self.b0n = sp.simplify(b(alpha, beta, n+1, n) * (h(alpha, beta, n+1, 1) / h(alpha, beta, n, 0)))
        self.b1n = None
        self.b2n = sp.simplify(b(alpha, beta, n+1, n+2) * (h(alpha, beta, n+1, 1)/ h(alpha, beta, n+2, 0)))
        if not self.alpha == self.beta:
            self.b1n = sp.simplify(b(alpha, beta, n+1, n+1) * (h(alpha, beta, n+1, 1)/ h(alpha, beta, n+1, 0)))

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'CD'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        k = np.arange(N)
        d0, d2 = np.zeros(N), np.zeros(N-2)
        d0[:-2] = sp.lambdify(n, self.b0n)(k[:N-2])
        d2[:] = sp.lambdify(n, self.b2n)(k[:N-2])
        if self.b1n:
            d1 = np.zeros(N-1)
            d1[:-1] = sp.lambdify(n, self.b1n)(k[:N-2])
            return SparseMatrix({0: d0, 1: d1, 2: d2}, (N, N))
        return SparseMatrix({0: d0, 2: d2}, (N, N))

    def slice(self):
        return slice(0, self.N-2)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCDirichlet(self.N, quad=self.quad, domain=self.domain,
                                     scaled=self._scaled, alpha=self.alpha, beta=self.beta,
                                     coordinates=self.coors.coordinates)
        return self._bc_basis

class ShenDirichlet(JacobiBase):
    """Jacobi function space for Dirichlet boundary conditions

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - JG - Jacobi-Gauss

        bc : tuple of numbers
            Boundary conditions at edges of domain
        domain : 2-tuple of floats, optional
            The computational domain
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))

    """
    def __init__(self, N, quad='JG', bc=(0, 0), domain=(-1., 1.), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None, alpha=-1, beta=-1):
        assert alpha == -1 and beta == -1
        JacobiBase.__init__(self, N, quad=quad, alpha=-1, beta=-1, domain=domain, dtype=dtype,
                            padding_factor=padding_factor, dealias_direct=dealias_direct,
                            coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'SD'

    def is_scaled(self):
        return False

    def slice(self):
        return slice(0, self.N-2)

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mpmath_points_and_weights()[0]
        N = self.shape(False)
        V = np.zeros((x.shape[0], N))
        for i in range(N-2):
            V[:, i] = self.evaluate_basis_derivative(x, i, k, output_array=V[:, i])
        return V

    def sympy_basis(self, i=0, x=xp):
        return (1-x**2)*sp.jacobi(i, 1, 1, x)
        #return (1-x)**(-self.alpha)*(1+x)**(-self.beta)*sp.jacobi(i, -self.alpha, -self.beta, x)

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mpmath_points_and_weights()[0]
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        f = self.sympy_basis(i, xp)
        output_array[:] = sp.lambdify(xp, f.diff(xp, k), mode)(x)
        return output_array

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        if mode == 'numpy':
            output_array = (1-x**2)*eval_jacobi(i, -self.alpha, -self.beta, x, out=output_array)
        else:
            f = self.sympy_basis(i, xp)
            output_array[:] = sp.lambdify(xp, f, 'mpmath')(x)
        return output_array

    def evaluate_basis_all(self, x=None, argument=0):
        if mode == 'numpy':
            if x is None:
                x = self.mesh(False, False)
            V = np.zeros((x.shape[0], self.N))
            V[:, :-2] = self.jacobi(x, 1, 1, self.N-2)*(1-x**2)[:, np.newaxis]
        else:
            if x is None:
                x = self.mpmath_points_and_weights()[0]
            N = self.shape(False)
            V = np.zeros((x.shape[0], N))
            for i in range(N-2):
                V[:, i] = self.evaluate_basis(x, i, output_array=V[:, i])
        return V

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)
        assert self.quad == "JG"
        points, weights = roots_jacobi(N, self.alpha+1, self.beta+1)
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def mpmath_points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if mode == 'numpy' or not has_quadpy:
            return self.points_and_weights(N=N, map_true_domain=map_true_domain, weighted=weighted, **kw)
        if N is None:
            N = self.shape(False)
        assert self.quad == "JG"
        pw = quadpy.c1.gauss_jacobi(N, self.alpha+1, self.beta+1, mode)
        points = pw.points_symbolic
        weights = pw.weights_symbolic
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def to_ortho(self, input_array, output_array=None):
        assert self.alpha == -1 and self.beta == -1
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)
        k = self.wavenumbers().astype(float)
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(2, None)]
        z = input_array[s0]*2*(k+1)/(2*k+3)
        output_array[s0] = z
        output_array[s1] -= z
        return output_array

    def _evaluate_scalar_product(self, fast_transform=True):
        SpectralBase._evaluate_scalar_product(self)
        self.scalar_product.tmp_array[self.sl[slice(-2, None)]] = 0

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCDirichlet(self.N, quad=self.quad, domain=self.domain,
                                     alpha=1, beta=1, coordinates=self.coors.coordinates)
        return self._bc_basis

class ShenBiharmonic(JacobiBase):
    """Function space for Biharmonic boundary conditions

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - JG - Jacobi-Gauss

        domain : 2-tuple of floats, optional
            The computational domain
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))

    Note
    ----
    The generalized Jacobi function j^{alpha=-2, beta=-2} is used as basis. However,
    inner products are computed without weights, for alpha=beta=0.

    """
    def __init__(self, N, quad='JG', bc=(0, 0, 0, 0), domain=(-1., 1.), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None,
                 alpha=-2, beta=-2):
        assert alpha == -2 and beta == -2
        JacobiBase.__init__(self, N, quad=quad, alpha=-2, beta=-2, domain=domain, dtype=dtype,
                            padding_factor=padding_factor, dealias_direct=dealias_direct,
                            coordinates=coordinates)
        assert bc in ((0, 0, 0, 0), 'Biharmonic')
        from shenfun.tensorproductspace import BoundaryValues
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    @staticmethod
    def short_name():
        return 'SB'

    def slice(self):
        return slice(0, self.N-4)

    def sympy_basis(self, i=0, x=xp):
        return (1-x**2)**2*sp.jacobi(i, 2, 2, x)

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mpmath_points_and_weights()[0]
        N = self.shape(False)
        V = np.zeros((x.shape[0], N))
        for i in range(N-4):
            V[:, i] = self.evaluate_basis_derivative(x, i, k, output_array=V[:, i])
        return V

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mpmath_points_and_weights()[0]
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        f = self.sympy_basis(i, xp)
        output_array[:] = sp.lambdify(xp, f.diff(xp, k), mode)(x)
        return output_array

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if mode == 'numpy':
            output_array[:] = (1-x**2)**2*eval_jacobi(i, 2, 2, x, out=output_array)
        else:
            f = self.sympy_basis(i, xp)
            output_array[:] = sp.lambdify(xp, f, 'mpmath')(x)
        return output_array

    def evaluate_basis_all(self, x=None, argument=0):
        if mode == 'numpy':
            if x is None:
                x = self.mesh(False, False)
            N = self.shape(False)
            V = np.zeros((x.shape[0], N))
            V[:, :-4] = self.jacobi(x, 2, 2, N-4)*((1-x**2)**2)[:, np.newaxis]
        else:
            if x is None:
                x = self.mpmath_points_and_weights()[0]
            N = self.shape(False)
            V = np.zeros((x.shape[0], N))
            for i in range(N-4):
                V[:, i] = self.evaluate_basis(x, i, output_array=V[:, i])
        return V

    def _evaluate_scalar_product(self, fast_transform=True):
        SpectralBase._evaluate_scalar_product(self)
        self.scalar_product.tmp_array[self.sl[slice(-4, None)]] = 0

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)
        assert self.quad == "JG"
        points, weights = roots_jacobi(N, 0, 0)
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def mpmath_points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if mode == 'numpy' or not has_quadpy:
            return self.points_and_weights(N=N, map_true_domain=map_true_domain, weighted=weighted, **kw)
        if N is None:
            N = self.shape(False)
        assert self.quad == "JG"
        pw = quadpy.c1.gauss_jacobi(N, 0, 0, 'mpmath')
        points = pw.points_symbolic
        weights = pw.weights_symbolic
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)
        k = self.wavenumbers().astype(float)
        _factor0 = 4*(k+2)*(k+1)/(2*k+5)/(2*k+3)
        _factor1 = (-2*(2*k+5)/(2*k+7))
        _factor2 = ((2*k+3)/(2*k+7))
        s0 = self.sl[slice(0, -4)]
        z = _factor0*input_array[s0]
        output_array[s0] = z
        output_array[self.sl[slice(2, -2)]] += z*_factor1
        output_array[self.sl[slice(4, None)]] += z*_factor2
        return output_array

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCBiharmonic(self.N, quad=self.quad, domain=self.domain,
                                      alpha=self.alpha, beta=self.beta, coordinates=self.coors.coordinates)
        return self._bc_basis


class ShenOrder6(JacobiBase):
    """Function space for 6th order equation

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - JG - Jacobi-Gauss

        domain : 2-tuple of floats, optional
            The computational domain
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))

    Note
    ----
    The generalized Jacobi function j^{alpha=-3, beta=-3} is used as basis. However,
    inner products are computed without weights, for alpha=beta=0.

    """
    def __init__(self, N, quad='JG', domain=(-1., 1.), dtype=float, padding_factor=1, dealias_direct=False,
                 coordinates=None, bc=(0, 0, 0, 0, 0, 0), alpha=-3, beta=-3):
        assert alpha == -3 and beta == -3
        JacobiBase.__init__(self, N, quad=quad, alpha=-3, beta=-3, domain=domain, dtype=dtype,
                            padding_factor=padding_factor, dealias_direct=dealias_direct,
                            coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return '6th order'

    @staticmethod
    def short_name():
        return 'SS'

    def slice(self):
        return slice(0, self.N-6)

    def sympy_basis(self, i=0, x=xp):
        return (1-x**2)**3*sp.jacobi(i, 3, 3, x)

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mpmath_points_and_weights()[0]
        if output_array is None:
            output_array = np.zeros(x.shape)
        f = self.sympy_basis(i, xp)
        output_array[:] = sp.lambdify(xp, f.diff(xp, k), mode)(x)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mpmath_points_and_weights()[0]
        N = self.shape(False)
        V = np.zeros((x.shape[0], N))
        for i in range(N-6):
            V[:, i] = self.evaluate_basis_derivative(x, i, k, output_array=V[:, i])
        return V

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if mode == 'numpy':
            output_array[:] = (1-x**2)**3*eval_jacobi(i, 3, 3, x, out=output_array)
        else:
            f = self.sympy_basis(i, xp)
            output_array[:] = sp.lambdify(xp, f, 'mpmath')(x)
        return output_array

    def evaluate_basis_all(self, x=None, argument=0):
        if mode == 'numpy':
            if x is None:
                x = self.mesh(False, False)
            N = self.shape(False)
            V = np.zeros((x.shape[0], N))
            V[:, :-6] = self.jacobi(x, 3, 3, N-6)*((1-x**2)**3)[:, np.newaxis]
        else:
            if x is None:
                x = self.mpmath_points_and_weights()[0]
            N = self.shape(False)
            V = np.zeros((x.shape[0], N))
            for i in range(N-6):
                V[:, i] = self.evaluate_basis(x, i, output_array=V[:, i])
        return V

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)
        assert self.quad == "JG"
        points, weights = roots_jacobi(N, 0, 0)
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def mpmath_points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if mode == 'numpy' or not has_quadpy:
            return self.points_and_weights(N=N, map_true_domain=map_true_domain, weighted=weighted, **kw)
        if N is None:
            N = self.shape(False)
        assert self.quad == "JG"
        pw = quadpy.c1.gauss_jacobi(N, 0, 0, 'mpmath')
        points = pw.points_symbolic
        weights = pw.weights_symbolic
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, weights

    def get_orthogonal(self):
        return Orthogonal(self.N, alpha=0, beta=0, dtype=self.dtype, domain=self.domain, coordinates=self.coors.coordinates)

    def _evaluate_scalar_product(self, fast_transform=True):
        SpectralBase._evaluate_scalar_product(self)
        self.scalar_product.tmp_array[self.sl[slice(-6, None)]] = 0

    #def to_ortho(self, input_array, output_array=None):
    #    if output_array is None:
    #        output_array = np.zeros_like(input_array.v)
    #    k = self.wavenumbers().astype(float)
    #    _factor0 = 4*(k+2)*(k+1)/(2*k+5)/(2*k+3)
    #    _factor1 = (-2*(2*k+5)/(2*k+7))
    #    _factor2 = ((2*k+3)/(2*k+7))
    #    s0 = self.sl[slice(0, -4)]
    #    z = _factor0*input_array[s0]
    #    output_array[s0] = z
    #    output_array[self.sl[slice(2, -2)]] -= z*_factor1
    #    output_array[self.sl[slice(4, None)]] += z*_factor2
    #    return output_array

class BCBase(CompositeSpace):
    """Function space for inhomogeneous boundary conditions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - JG - Jacobi-Gauss

        domain : 2-tuple of floats, optional
            The computational domain
        scaled : bool, optional
            Whether or not to use scaled basis
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """

    def __init__(self, N, quad="JG", domain=(-1., 1.), scaled=False,
                 dtype=float, alpha=0, beta=0, coordinates=None, **kw):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain,
                                dtype=dtype, alpha=alpha, beta=beta,
                                coordinates=coordinates)

    def stencil_matrix(self, N=None):
        raise NotImplementedError

    @staticmethod
    def short_name():
        raise NotImplementedError

    @staticmethod
    def boundary_condition():
        return 'Apply'

    def shape(self, forward_output=True):
        if forward_output:
            return self.stencil_matrix().shape[0]
        else:
            return self.N

    @property
    def num_T(self):
        return self.stencil_matrix().shape[1]

    def slice(self):
        return slice(self.N-self.shape(), self.N)

    def vandermonde(self, x):
        return self.jacobi(x, self.alpha, self.beta, self.num_T)
        #return Orthogonal.vandermonde(self, x)

    def _composite(self, V, argument=1):
        N = self.shape()
        P = np.zeros(V[:, :N].shape)
        P[:] = np.tensordot(V[:, :self.num_T], self.stencil_matrix(), (1, 1))
        return P

    def sympy_basis(self, i=0, x=xp):
        M = self.stencil_matrix()
        return np.sum(M[i]*np.array([sp.jacobi(j, self.alpha, self.beta, x) for j in range(self.num_T)]))

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        V = self.vandermonde(x)
        output_array[:] = np.dot(V, self.stencil_matrix()[i])
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        output_array = SpectralBase.evaluate_basis_derivative(self, x=x, i=i, k=k, output_array=output_array)
        return output_array

    def to_ortho(self, input_array, output_array=None):
        from shenfun import Function
        T = self.get_orthogonal()
        if output_array is None:
            output_array = Function(T)
        else:
            output_array.fill(0)
        M = self.stencil_matrix().T
        for k, row in enumerate(M):
            output_array[k] = np.dot(row, input_array)
        return output_array

    def eval(self, x, u, output_array=None):
        v = self.to_ortho(u)
        output_array = v.eval(x, output_array=output_array)
        return output_array

class BCDirichlet(BCBase):

    @staticmethod
    def short_name():
        return 'BCD'

    def stencil_matrix(self, N=None):
        a, b = sp.symbols('a,b', real=True)
        A = sp.Matrix([[(a + 1)/(a + b + 2), -1/(a + b + 2)],
                       [(b + 1)/(a + b + 2),  1/(a + b + 2)]])
        return np.array(A.subs(a, self.alpha).subs(b, self.beta))

class BCNeumann(BCBase):

    @staticmethod
    def short_name():
        return 'BCN'

    def stencil_matrix(self, N=None):
        a, b = sp.symbols('a,b', real=True)
        A = sp.Matrix([[(a + 2)/(a + b + 4), -1/(a + b + 4)],
                       [(b + 2)/(a + b + 4),  1/(a + b + 4)]])
        return np.array(A.subs(a, self.alpha).subs(b, self.beta))

class BCBiharmonic(BCBase):

    @staticmethod
    def short_name():
        return 'BCB'

    def stencil_matrix(self, N=None):
        return sp.Rational(1, 16)*np.array([[8, -9, 0, 1],
                                            [2, -1, -2, 1],
                                            [8, 9, 0, -1],
                                            [-2, -1, 2, 1]])
