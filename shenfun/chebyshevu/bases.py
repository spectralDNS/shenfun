"""
Module for defining function spaces in the Chebyshev family
"""
from __future__ import division
import functools
import numpy as np
import sympy as sp
from time import time
from scipy.special import eval_chebyu
from numpy.polynomial import chebyshev as n_cheb
from mpi4py_fft import fftw
from shenfun.spectralbase import SpectralBase, Transform, FuncWrap, \
    islicedict, slicedict
from shenfun.matrixbase import SparseMatrix
from shenfun.config import config

#pylint: disable=abstract-method, not-callable, method-hidden, no-self-use, cyclic-import

xp = sp.Symbol('x', real=True)

class DCTWrap(FuncWrap):
    """DCT for complex input"""

    @property
    def dct(self):
        return object.__getattribute__(self, '_func')

    def __call__(self, input_array=None, output_array=None, **kw):
        dct_obj = self.dct

        if input_array is not None:
            self.input_array[...] = input_array

        dct_obj.input_array[...] = self.input_array.real
        dct_obj(None, None, **kw)
        self.output_array.real[...] = dct_obj.output_array
        dct_obj.input_array[...] = self.input_array.imag
        dct_obj(None, None, **kw)
        self.output_array.imag[...] = dct_obj.output_array

        if output_array is not None:
            output_array[...] = self.output_array
            return output_array
        return self.output_array

class ChebyshevuBase(SpectralBase):

    @staticmethod
    def family():
        return 'chebyshevu'

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)

        if weighted:
            if self.quad == "GC":
                points, weights = n_cheb.chebgauss(N)
                points = points.astype(float)
                weights = (weights*(1-points**2)).astype(float)

            elif self.quad == "GU":
                points = np.cos((np.arange(N)+1)*np.pi/(N+1))
                weights = np.full(N, np.pi/(N+1))*(1-points**2)

        else:
            if self.quad == "GU":
                import quadpy
                p = quadpy.c1.clenshaw_curtis(N)
                points = -p.points
                weights = p.weights

            elif self.quad == "GC":
                points = n_cheb.chebgauss(N)[0]
                d = fftw.aligned(N, fill=0)
                k = 2*(1 + np.arange((N-1)//2))
                d[::2] = (2./N)/np.hstack((1., 1.-k*k))
                w = fftw.aligned_like(d)
                dct = fftw.dctn(w, axes=(0,), type=3)
                weights = dct(d, w)

        if map_true_domain is True:
            points = self.map_true_domain(points)

        return points, weights

    def vandermonde(self, x):
        return chebvanderU(x, self.shape(False)-1)

    def weight(self, x=xp):
        return sp.sqrt(1-x**2)

    def reference_domain(self):
        return (-1, 1)

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

        U = self._U
        V = self._V
        trunc_array = self._tmp

        self.axis = axis
        if self.padding_factor != 1:
            self.scalar_product = Transform(self.scalar_product, self.xfftn_fwd, U, V, trunc_array)
            self.forward = Transform(self.forward, self.xfftn_fwd, U, V, trunc_array)
            self.backward = Transform(self.backward, self.xfftn_bck, trunc_array, V, U)
        else:
            self.scalar_product = Transform(self.scalar_product, self.xfftn_fwd, U, V, V)
            self.forward = Transform(self.forward, self.xfftn_fwd, U, V, V)
            self.backward = Transform(self.backward, self.xfftn_bck, V, V, U)
        self.si = islicedict(axis=self.axis, dimensions=U.ndim)
        self.sl = slicedict(axis=self.axis, dimensions=U.ndim)

    def get_orthogonal(self):
        return Orthogonal(self.N, quad=self.quad, domain=self.domain, dtype=self.dtype,
                          padding_factor=self.padding_factor,
                          dealias_direct=self.dealias_direct,
                          coordinates=self.coors.coordinates)

class Orthogonal(ChebyshevuBase):
    """Function space for Chebyshev series of second kind

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GC - Chebyshev-Gauss
            - GU - Chebyshevu-Gauss
        domain : 2-tuple of floats, optional
            The computational domain
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))
    """

    def __init__(self, N, quad='GU', domain=(-1, 1), dtype=float, padding_factor=1,
                 dealias_direct=False, coordinates=None):
        ChebyshevuBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)
        assert quad in ('GU', 'GC')
        if quad == 'GC':
            self._xfftn_fwd = functools.partial(fftw.dstn, type=2)
            self._xfftn_bck = functools.partial(fftw.dstn, type=3)
            self._sinGC = np.sin((np.arange(N)+0.5)*np.pi/N)

        elif quad == 'GU':
            self._xfftn_fwd = functools.partial(fftw.dstn, type=1)
            self._xfftn_bck = functools.partial(fftw.dstn, type=1)
        self._xfftn_fwd.opts = self._xfftn_bck.opts = config['fftw']['dst']
        self.plan((int(padding_factor*N),), 0, dtype, {})

    def sympy_basis(self, i=0, x=xp):
        return sp.chebyshevu(i, x)

    def bnd_values(self, k=0):
        if k == 0:
            return (lambda i: (-1)**i*(i+1), lambda i: (i+1))
        elif k == 1:
            return (lambda i: sp.Rational(1, 3)*((i+1)**3-(i+1)), lambda i: sp.Rational(1, 3)(-1)**(i+1)*((i+1)**3-(i+1)) )

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array[:] = eval_chebyu(i, x)
        return output_array

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            x = self.mesh(False, False)
        return self.vandermonde(x)

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        # Implementing using U_k = \frac{1}{k+1}T^{'}_{k+1}
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True)+1)
        basis[i+1] = 1
        basis = n_cheb.Chebyshev(basis)
        k += 1
        basis = basis.deriv(k)
        output_array[:] = basis(x)/(i+1)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        # Implementing using  U_k = \frac{1}{k+1}T^{'}_{k+1}
        if x is None:
            x = self.mesh(False, False)
        V = n_cheb.chebvander(x, self.shape(False))
        M = V.shape[1]
        D = np.zeros((M, M))
        k = k+1
        D[slice(0, M-k)] = n_cheb.chebder(np.eye(M, M), k)
        V = np.dot(V, D)
        i = np.arange(1, M)[None, :]
        return V[:, 1:]/i

    def _evaluate_scalar_product(self, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            return

        if self.quad == 'GU':
            self.scalar_product._input_array *= self.broadcast_to_ndims(np.sin(np.pi/(self.N+1)*(np.arange(1, self.N+1))))
            out = self.scalar_product.xfftn()
            out *= (np.pi/(2*(self.N+1)*self.padding_factor*self.domain_factor()))

        elif self.quad == 'GC':
            self.scalar_product._input_array *= self.broadcast_to_ndims(self._sinGC)
            out = self.scalar_product.xfftn()
            out *= (np.pi/(2*self.N*self.padding_factor*self.domain_factor()))

    def _evaluate_expansion_all(self, input_array, output_array, x=None, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_expansion_all(self, input_array, output_array, x, False)
            return

        # Fast transform. Make sure arrays are correct. Fast transforms are only for planned size
        assert input_array is self.backward.tmp_array
        assert output_array is self.backward.output_array
        output_array = self.backward.xfftn()

        if self.quad == "GU":
            output_array *= 1/(2*self.broadcast_to_ndims(np.sin((np.arange(self.N)+1)*np.pi/(self.N+1))))

        elif self.quad == "GC":
            s0 = self.sl[slice(self.N-1, self.N)]
            se = self.sl[slice(0, self.N, 2)]
            so = self.sl[slice(1, self.N, 2)]
            output_array *= 0.5
            output_array[se] += input_array[s0]/2
            output_array[so] -= input_array[s0]/2
            output_array *= 1/(self.broadcast_to_ndims(np.sin((np.arange(self.N)+0.5)*np.pi/self.N)))

    @property
    def is_orthogonal(self):
        return True

    @staticmethod
    def short_name():
        return 'U'

    def to_ortho(self, input_array, output_array=None):
        assert input_array.__class__.__name__ == 'Orthogonal'
        if output_array:
            output_array[:] = input_array
            return output_array
        return input_array

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.forward.output_array.dtype)
        x = self.map_reference_domain(x)
        output_array[:] = np.dot(chebvanderU(x, self.N-1), u)
        return output_array

    def stencil_matrix(self, N=None):
        """Matrix describing the linear combination of orthogonal basis
        functions for the current basis.

        Parameters
        ----------
        N : int, optional
            The number of quadrature points
        """
        raise SparseMatrix({0: 1}, (N, N))

    def plan(self, shape, axis, dtype, options):
        if shape in (0, (0,)):
            return

        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[-1]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        plan_fwd = self._xfftn_fwd
        plan_bck = self._xfftn_bck

        opts = plan_fwd.opts
        opts['overwrite_input'] = 'FFTW_DESTROY_INPUT'
        opts.update(options)
        flags = (fftw.flag_dict[opts['planner_effort']],
                 fftw.flag_dict[opts['overwrite_input']])
        threads = opts['threads']

        U = fftw.aligned(shape, dtype=float)

        xfftn_fwd = plan_fwd(U, axes=(axis,), threads=threads, flags=flags)
        V = xfftn_fwd.output_array
        xfftn_bck = plan_bck(V, axes=(axis,), threads=threads, flags=flags, output_array=U)
        V.fill(0)
        U.fill(0)

        if np.dtype(dtype) is np.dtype('complex'):
            # dct only works on real data, so need to wrap it
            U = fftw.aligned(shape, dtype=complex)
            V = fftw.aligned(shape, dtype=complex)
            U.fill(0)
            V.fill(0)
            xfftn_fwd = DCTWrap(xfftn_fwd, U, V)
            xfftn_bck = DCTWrap(xfftn_bck, V, U)

        self.axis = axis
        if self.__class__.__name__ == 'Orthogonal':
            if self.padding_factor != 1:
                trunc_array = self._get_truncarray(shape, V.dtype)
                self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, trunc_array)
                self.forward = Transform(self.forward, xfftn_fwd, U, V, trunc_array)
                self.backward = Transform(self.backward, xfftn_bck, trunc_array, V, U)
            else:
                self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)
                self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
                self.backward = Transform(self.backward, xfftn_bck, V, V, U)

        else:
            self.xfftn_fwd = xfftn_fwd
            self.xfftn_bck = xfftn_bck
            self._U = U
            self._V = V
            if self.padding_factor != 1:
                trunc_array = self._get_truncarray(shape, V.dtype)
                self._tmp = trunc_array
            else:
                self._tmp = V

        self.si = islicedict(axis=self.axis, dimensions=U.ndim)
        self.sl = slicedict(axis=self.axis, dimensions=U.ndim)

class CompositeSpace(Orthogonal):
    """Common abstract class for all spaces based on composite bases of Chebyshev
    polynomials of second kind."""

    def __init__(self, N, quad="GU", bc=(0, 0), domain=(-1., 1.), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        Orthogonal.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                             padding_factor=padding_factor, dealias_direct=dealias_direct,
                             coordinates=coordinates)
        ChebyshevuBase.plan(self, (int(padding_factor*N),), 0, dtype, {})
        from shenfun.tensorproductspace import BoundaryValues
        self._scaled = scaled
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    def plan(self, shape, axis, dtype, options):
        Orthogonal.plan(self, shape, axis, dtype, options)
        ChebyshevuBase.plan(self, shape, axis, dtype, options)

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
        Orthogonal._evaluate_expansion_all(self, input_array, output_array, x, True)

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
    def is_scaled(self):
        """Return True if scaled basis is used, otherwise False"""
        return self._scaled

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

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
                f += val*Orthogonal.sympy_basis(self, i=j, x=x)
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

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w = self.to_ortho(u)
        output_array[:] = Orthogonal.eval(self, x, w)
        return output_array

class Phi1(CompositeSpace):
    r"""Function space for Dirichlet boundary conditions

    .. math::

        u(x_0)=bc[0] \text{ and } u(x_1)=bc[1], \quad \text{for } x \in [x_0, x_1]

    The basis function is

    .. math::

        \phi_n &= \frac{1}{\pi}\left(\frac{U_n}{n+1}-\frac{U_{n+2}}{n+3}\right) \\
               &=  \frac{(1-x^2)U'_{k+1}}{h^{(1)}_{k+1}}

    where :math:`h^{(1)}_n = \frac{\pi n(n+2)}{2}`.

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

        bc : 2-tuple of floats, optional
            Boundary conditions at, respectively, x=(-1, 1).
        domain : 2-tuple of floats, optional
            The computational domain
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        scaled : bool, optional
            Whether or not to use scaled basis
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))

    Note
    ----
    A test function is always using homogeneous boundary conditions.

    """
    def __init__(self, N, quad="GU", bc=(0., 0.), domain=(-1., 1.), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)
        from shenfun.jacobi.recursions import b, h, half, n, un
        self.b0n = sp.simplify(b(half, half, n+1, n, un) / (h(half, half, n, 0, un)))
        self.b2n = sp.simplify(b(half, half, n+1, n+2, un) / (h(half, half, n+2, 0, un)))

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'P1'

    def stencil_matrix(self, N=None):
        from shenfun.jacobi.recursions import n
        N = self.N if N is None else N
        k = np.arange(N)
        d0, d2 = np.zeros(N), np.zeros(N-2)
        d0[:-2] = sp.lambdify(n, self.b0n)(k[:N-2])
        d2[:] = sp.lambdify(n, self.b2n)(k[:N-2])
        if self.is_scaled:
            return SparseMatrix({0: d0/(k+2), 2: d2/(k[:-2]+2)}, (N, N))
        return SparseMatrix({0: d0, 2: d2}, (N, N))

    def slice(self):
        return slice(0, self.N-2)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCDirichlet(self.N, quad=self.quad, domain=self.domain,
                                     scaled=self._scaled, coordinates=self.coors.coordinates)
        return self._bc_basis

class Phi2(CompositeSpace):
    r"""Function space for Biharmonic boundary conditions

    u(-1)=a, u'(-1)=b, u(1)=c and u'(1)=d

    The basis function is

    .. math::

        \phi_n &= \frac{(1-x^2)^2 U''_{k+2}}{h^{(2)}_{k+2}} \\
               &= \frac{1}{2\pi(n+1)(n+2)}\left(U_n- \frac{2(n+1)}{n+4}U_{n+2} + \frac{(n+1)(n+2)}{(n+3)(n+4)}U_{n+4} \right)

    where :math:`h^{(2)}_n = \frac{\pi (n+3)(n+2)n(n-1)}{2}`.

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

        bc : 4-tuple of floats, optional
            2 boundary conditions at, respectively, x=-1 and x=1.
        domain : 2-tuple of floats, optional
            The computational domain
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        scaled : bool, optional
            Whether or not to use scaled basis
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))

    Note
    ----
    A test function is always using homogeneous boundary conditions.

    """
    def __init__(self, N, quad="GU", bc=(0, 0, 0, 0), domain=(-1, 1), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)
        from shenfun.jacobi.recursions import b, h, half, n, matpow, un
        self.b0n = sp.simplify(matpow(b, 2, half, half, n+2, n, un) / (h(half, half, n, 0, un)))
        self.b2n = sp.simplify(matpow(b, 2, half, half, n+2, n+2, un) / (h(half, half, n+2, 0, un)))
        self.b4n = sp.simplify(matpow(b, 2, half, half, n+2, n+4, un) / (h(half, half, n+4, 0, un)))

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    @staticmethod
    def short_name():
        return 'P2'

    def stencil_matrix(self, N=None):
        from shenfun.jacobi.recursions import n
        N = self.N if N is None else N
        k = np.arange(N)
        d0, d2, d4 = np.zeros(N), np.zeros(N-2), np.zeros(N-4)
        d0[:-4] = sp.lambdify(n, self.b0n)(k[:N-4])
        d2[:-2] = sp.lambdify(n, self.b2n)(k[:N-4])
        d4[:] = sp.lambdify(n, self.b4n)(k[:N-4])
        return SparseMatrix({0: d0, 2: d2, 4: d4}, (N, N))

    def slice(self):
        return slice(0, self.N-4)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCBiharmonic(self.N, quad=self.quad, domain=self.domain,
                                      scaled=self._scaled, coordinates=self.coors.coordinates)
        return self._bc_basis

class CompactDirichlet(CompositeSpace):
    r"""Function space for Dirichlet boundary conditions

    u(-1)=a and u(1)=b

    The basis function is

    .. math::

        \phi_n = {U_n}-\frac{n+1}{n+3} U_{n+2}

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

        bc : 2-tuple of floats, optional
            Boundary conditions at, respectively, x=(-1, 1).
        domain : 2-tuple of floats, optional
            The computational domain
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        scaled : bool, optional
            Whether or not to use scaled basis
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))

    Note
    ----
    A test function is always using homogeneous boundary conditions.
    This basis function is a scaled version of :class:`Phi1`.

    """
    def __init__(self, N, quad="GU", bc=(0, 0), domain=(-1, 1), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)

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
        d0[:-2] = 1
        d2[:] = -(k[:-2]+1)/(k[:-2]+3)
        return SparseMatrix({0: d0, 2: d2}, (N, N))

    def slice(self):
        return slice(0, self.N-2)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCDirichlet(self.N, quad=self.quad, domain=self.domain,
                                     scaled=self._scaled, coordinates=self.coors.coordinates)
        return self._bc_basis

class CompactNeumann(CompositeSpace):
    r"""Function space for Neumann boundary conditions

    u'(-1)=a and u'(1)=b

    The basis function is

    .. math::

        \phi_n = {U_n}-\frac{n (n+1)}{(n+3)(n+4)} U_{n+2}

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

        bc : 2-tuple of floats, optional
            Boundary conditions at, respectively, x=(-1, 1).
        domain : 2-tuple of floats, optional
            The computational domain
        dtype : data-type, optional
            Type of input data in real physical space. Will be overloaded when
            basis is part of a :class:`.TensorProductSpace`.
        scaled : bool, optional
            Whether or not to use scaled basis
        padding_factor : float, optional
            Factor for padding backward transforms.
        dealias_direct : bool, optional
            Set upper 1/3 of coefficients to zero before backward transform
        coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
            Map for curvilinear coordinatesystem.
            The new coordinate variable in the new coordinate system is the first item.
            Second item is a tuple for the Cartesian position vector as function of the
            new variable in the first tuple. Example::

                theta = sp.Symbols('x', real=True, positive=True)
                rv = (sp.cos(theta), sp.sin(theta))

    Note
    ----
    A test function is always using homogeneous boundary conditions.

    """
    def __init__(self, N, quad="GU", bc=(0, 0), domain=(-1, 1), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Neumann'

    @staticmethod
    def short_name():
        return 'CN'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        k = np.arange(N)
        d0, d2 = np.zeros(N), np.zeros(N-2)
        d0[:-2] = 1
        d2[:] = -k[:-2]*(k[:-2]+1)/((k[:-2]+3)*(k[:-2]+4))
        return SparseMatrix({0: d0, 2: d2}, (N, N))

    def slice(self):
        return slice(0, self.N-2)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCNeumann(self.N, quad=self.quad, domain=self.domain,
                                   scaled=self._scaled, coordinates=self.coors.coordinates)
        return self._bc_basis


class BCBase(CompositeSpace):
    """Function space for inhomogeneous boundary conditions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

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

    def __init__(self, N, quad="GU", domain=(-1., 1.), scaled=False,
                 dtype=float, coordinates=None, **kw):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain,
                                dtype=dtype, coordinates=coordinates)

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
        return chebvanderU(x, self.num_T-1)

    def _composite(self, V, argument=1):
        N = self.shape()
        P = np.zeros(V[:, :N].shape)
        P[:] = np.tensordot(V[:, :self.num_T], self.stencil_matrix(), (1, 1))
        return P

    def sympy_basis(self, i=0, x=xp):
        M = self.stencil_matrix()
        return np.sum(M[i]*np.array([sp.chebyshevu(j, x) for j in range(self.num_T)]))

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
    r"""Basis for inhomogeneous Dirichlet boundary conditions

    .. math::

        \phi_0 &= \frac{1}{2}U_0 - \frac{1}{4}U_1 \\
        \phi_1 &= \frac{1}{2}U_0 + \frac{1}{4}U_1
    """
    @staticmethod
    def short_name():
        return 'BCD'

    def stencil_matrix(self, N=None):
        return sp.Rational(1, 4)*np.array([[2, -1],
                                           [2,  1]])

class BCNeumann(BCBase):
    r"""Basis for inhomogeneous Neumann boundary conditions

    .. math::

        \phi_0 &= \frac{1}{16}(4U_1 - U_2) \\
        \phi_1 &= \frac{1}{16}(4U_1 + U_2)
    """

    @staticmethod
    def short_name():
        return 'BCN'

    def stencil_matrix(self, N=None):
        return sp.Rational(1, 16)*np.array([[0, 4, -1],
                                            [0, 4, 1]])

class BCBiharmonic(BCBase):
    r"""Basis for inhomogeneous Biharmonic boundary conditions

    .. math::

        \phi_0 &= \frac{1}{32}(16U_0 - 10U_1 + U_3 \\
        \phi_1 &= \frac{1}{32}(6U_0 - 2U_1 - 2U_2 + U_3) \\
        \phi_2 &= \frac{1}{32}(16U_0 + 10U_1 - U_3) \\
        \phi_3 &= \frac{1}{32}(-6U_0 - 2U_1 + 2U_2 + U_3)
    """

    @staticmethod
    def short_name():
        return 'BCB'

    def stencil_matrix(self, N=None):
        return sp.Rational(1, 32)*np.array([[16, -10,  0,  1],
                                            [ 6,  -2, -2,  1],
                                            [16,  10,  0, -1],
                                            [-6,  -2,  2,  1]])

def chebvanderU(x, deg):
    """Pseudo-Vandermonde matrix of given degree for Chebyshev polynomials
    of second kind.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = U_i(x),

    where `0 <= i <= deg`. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the Chebyshev polynomial.

    If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the
    matrix ``V = chebvander(x, n)``, then ``np.dot(V, c)`` and
    ``chebval(x, c)`` are the same up to roundoff.  This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of Chebyshev series of the same degree and sample points.

    Parameters
    ----------
    x : array_like
        Array of points. The dtype is converted to float64 or complex128
        depending on whether any of the elements are complex. If `x` is
        scalar it is converted to a 1-D array.
    deg : int
        Degree of the resulting matrix.

    Returns
    -------
    vander : ndarray
        The pseudo Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where The last index is the degree of the
        corresponding Chebyshev polynomial.  The dtype will be the same as
        the converted `x`.

    """
    import numpy.polynomial.polyutils as pu
    ideg = pu._deprecate_as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = np.array(x, copy=False, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = np.empty(dims, dtype=dtyp)
    # Use forward recursion to generate the entries.
    v[0] = x*0 + 1
    if ideg > 0:
        x2 = 2*x
        v[1] = 2*x
        for i in range(2, ideg + 1):
            v[i] = v[i-1]*x2 - v[i-2]
    return np.moveaxis(v, 0, -1)
