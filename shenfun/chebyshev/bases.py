"""
Module for defining function spaces in the Chebyshev family
"""
from __future__ import division
import functools
import numpy as np
import sympy as sp
from time import time
from numpy.polynomial import chebyshev as n_cheb
from scipy.special import eval_chebyt, eval_chebyu
from mpi4py_fft import fftw
from shenfun.spectralbase import SpectralBase, work, Transform, FuncWrap, \
    islicedict, slicedict
from shenfun.matrixbase import SparseMatrix
from shenfun.optimization import optimizer

__all__ = ['ChebyshevBase',
           'Orthogonal',
           'OrthogonalU',
           'ShenDirichlet',
           'DirichletU',
           'Heinrichs',
           'ShenNeumann',
           'CombinedShenNeumann',
           'MikNeumann',
           'ShenBiharmonic',
           'SecondNeumann',
           'UpperDirichlet',
           'UpperDirichletNeumann',
           'ShenBiPolar',
           'DirichletNeumann',
           'NeumannDirichlet',
           'BCDirichlet',
           'BCBiharmonic',
           'BCNeumann',
           'BCUpperDirichlet',
           'BCNeumannDirichlet',
           'BCDirichletNeumann',
           'BCUpperDirichletNeumann']


#pylint: disable=abstract-method, not-callable, method-hidden, no-self-use, cyclic-import

chebval = optimizer(n_cheb.chebval)

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


class ChebyshevBase(SpectralBase):

    @staticmethod
    def family():
        return 'chebyshev'

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)

        if weighted:
            if self.quad == "GL":
                points = -(n_cheb.chebpts2(N)).astype(float)
                weights = np.full(N, np.pi/(N-1))
                weights[0] /= 2
                weights[-1] /= 2

            elif self.quad == "GC":
                points, weights = n_cheb.chebgauss(N)
                points = points.astype(float)
                weights = weights.astype(float)

            elif self.quad == "GU":
                points = np.cos((np.arange(N)+1)*np.pi/(N+1))
                weights = np.full(N, np.pi/(N+1))

        else:
            if self.quad in ("GL", "GU"):
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
        return n_cheb.chebvander(x, self.shape(False)-1)

    def weight(self, x=xp):
        return 1/sp.sqrt(1-x**2)

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
            self.forward = Transform(self.forward, self.xfftn_fwd, U, V, trunc_array)
            self.backward = Transform(self.backward, self.xfftn_bck, trunc_array, V, U)
        else:
            self.forward = Transform(self.forward, self.xfftn_fwd, U, V, V)
            self.backward = Transform(self.backward, self.xfftn_bck, V, V, U)
        self.scalar_product = Transform(self.scalar_product, self.xfftn_fwd, U, V, V)
        self.si = islicedict(axis=self.axis, dimensions=U.ndim)
        self.sl = slicedict(axis=self.axis, dimensions=U.ndim)

    def get_orthogonal(self):
        return Orthogonal(self.N, quad=self.quad, domain=self.domain, dtype=self.dtype,
                          padding_factor=self.padding_factor,
                          dealias_direct=self.dealias_direct,
                          coordinates=self.coors.coordinates)

class Orthogonal(ChebyshevBase):
    """Function space for regular Chebyshev series

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

    def __init__(self, N, quad='GC', domain=(-1., 1.), dtype=float, padding_factor=1,
                 dealias_direct=False, coordinates=None):
        ChebyshevBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        if quad == 'GC':
            self._xfftn_fwd = functools.partial(fftw.dctn, type=2)
            self._xfftn_bck = functools.partial(fftw.dctn, type=3)

        else:
            self._xfftn_fwd = functools.partial(fftw.dctn, type=1)
            self._xfftn_bck = functools.partial(fftw.dctn, type=1)
        self.plan((int(padding_factor*N),), 0, dtype, {})

    # Comment due to curvilinear issues
    #def apply_inverse_mass(self, array):
    #    array *= (2/np.pi)
    #    array[self.si[0]] /= 2
    #    if self.quad == 'GL':
    #        array[self.si[-1]] /= 2
    #    return array

    def sympy_basis(self, i=0, x=xp):
        return sp.chebyshevt(i, x)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        #output_array[:] = np.cos(i*np.arccos(x))
        output_array[:] = eval_chebyt(i, x)
        return output_array

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            x = self.mesh(False, False)
        return self.vandermonde(x)

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[i] = 1
        basis = n_cheb.Chebyshev(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mesh(False, False)
        V = self.vandermonde(x)
        M = V.shape[1]
        if k > 0:
            D = np.zeros((M, M))
            D[:-k] = n_cheb.chebder(np.eye(M, M), k)
            V = np.dot(V, D)
        return V

    def _evaluate_expansion_all(self, input_array, output_array, x=None, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_expansion_all(self, input_array, output_array, x, False)
            return

        assert input_array is self.backward.tmp_array
        assert output_array is self.backward.output_array
        output_array = self.backward.xfftn()

        if self.quad == "GC":
            s0 = self.sl[slice(0, 1)]
            output_array *= 0.5
            output_array += input_array[s0]/2

        elif self.quad == "GL":
            output_array *= 0.5
            output_array += input_array[self.sl[slice(0, 1)]]/2
            s0 = self.sl[slice(-1, None)]
            s2 = self.sl[slice(0, None, 2)]
            output_array[s2] += input_array[s0]/2
            s2 = self.sl[slice(1, None, 2)]
            output_array[s2] -= input_array[s0]/2

    def _evaluate_scalar_product(self, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            return

        out = self.scalar_product.xfftn()

        if self.quad == "GC":
            out *= (np.pi/(2*self.N*self.padding_factor))

        elif self.quad == "GL":
            out *= (np.pi/(2*(self.N*self.padding_factor-1)))

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.forward.output_array.dtype)
        x = self.map_reference_domain(x)
        output_array[:] = chebval(x, u)
        #output_array[:] = n_cheb.chebval(x, u)
        return output_array

    @property
    def is_orthogonal(self):
        return True

    @staticmethod
    def short_name():
        return 'T'

    def to_ortho(self, input_array, output_array=None):
        assert input_array.__class__.__name__ == 'Orthogonal'
        if output_array:
            output_array[:] = input_array
            return output_array
        return input_array

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

        opts = dict(
            overwrite_input='FFTW_DESTROY_INPUT',
            planner_effort='FFTW_MEASURE',
            threads=1,
        )
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
                self.forward = Transform(self.forward, xfftn_fwd, U, V, trunc_array)
                self.backward = Transform(self.backward, xfftn_bck, trunc_array, V, U)
            else:
                self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
                self.backward = Transform(self.backward, xfftn_bck, V, V, U)
            self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)
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

# Note that all composite spaces rely on the fast transforms of
# the orthogonal space. For this reason we have an intermediate
# class CompositeSpace for all composite spaces, where common code
# is implemented and reused by all.

class OrthogonalU(ChebyshevBase):
    """Function space for Chebyshev series of second kind

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

    def __init__(self, N, quad='GC', domain=(-1., 1.), dtype=float, padding_factor=1,
                 dealias_direct=False, coordinates=None):
        ChebyshevBase.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                               padding_factor=padding_factor, dealias_direct=dealias_direct,
                               coordinates=coordinates)
        assert quad in ('GC', 'GL', 'GU')
        if quad == 'GC':
            self._xfftn_fwd = functools.partial(fftw.dstn, type=2)
            self._xfftn_bck = functools.partial(fftw.dstn, type=3)
            self._sinGC = np.sin((np.arange(N)+0.5)*np.pi/N)

        elif quad == 'GU':
            self._xfftn_fwd = functools.partial(fftw.dstn, type=1)
            self._xfftn_bck = functools.partial(fftw.dstn, type=1)

        self.plan((int(padding_factor*N),), 0, dtype, {})

    def sympy_basis(self, i=0, x=xp):
        return sp.chebyshevu(i, x)

    def vandermonde(self, x):
        return chebvanderU(x, self.shape(False)-1)

    # Using weight sqrt(1-x**2) modifies quadrature weights, but for now
    # all Chebyshev bases use weight 1/sqrt(1-x**2) by default, even the
    # second kind.

    #def weight(self, x=xp):
    #    return sp.sqrt(1-x**2)

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

        # Assuming weight is 1/sqrt(1-x**2). If this is changed to sqrt(1-x**2),
        # which is more natural for second kind, then use *= instead of /= below.
        if self.quad == 'GU':
            self.scalar_product._input_array /= self.broadcast_to_ndims(np.sin(np.pi/(self.N+1)*(np.arange(1, self.N+1))))
            out = self.scalar_product.xfftn()
            out *= (np.pi/(2*(self.N+1)*self.padding_factor))

        elif self.quad == 'GC':
            self.scalar_product._input_array /= self.broadcast_to_ndims(self._sinGC)
            out = self.scalar_product.xfftn()
            out *= (np.pi/(2*self.N*self.padding_factor))

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
        assert input_array.__class__.__name__ == 'OrthogonalU'
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


        opts = dict(
            overwrite_input='FFTW_DESTROY_INPUT',
            planner_effort='FFTW_MEASURE',
            threads=1,
        )
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
        if self.__class__.__name__ == 'OrthogonalU':
            if self.padding_factor != 1:
                trunc_array = self._get_truncarray(shape, V.dtype)
                self.forward = Transform(self.forward, xfftn_fwd, U, V, trunc_array)
                self.backward = Transform(self.backward, xfftn_bck, trunc_array, V, U)
            else:
                self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
                self.backward = Transform(self.backward, xfftn_bck, V, V, U)
            self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)
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
    """Common class for all spaces based on composite bases"""

    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1., 1.), dtype=float,
                 scaled=False, padding_factor=1, dealias_direct=False, coordinates=None):
        Orthogonal.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                            padding_factor=padding_factor, dealias_direct=dealias_direct,
                            coordinates=coordinates)
        ChebyshevBase.plan(self, (int(padding_factor*N),), 0, dtype, {})
        from shenfun.tensorproductspace import BoundaryValues
        self._bc_basis = None
        self._scaled = scaled
        self.bc = BoundaryValues(self, bc=bc)

    def plan(self, shape, axis, dtype, options):
        Orthogonal.plan(self, shape, axis, dtype, options)
        ChebyshevBase.plan(self, shape, axis, dtype, options)

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
        output = self.scalar_product.output_array
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            output[self.sl[slice(-(self.N-self.dim()), None)]] = 0
            return
        Orthogonal._evaluate_scalar_product(self, True)
        nbcs = self.N-self.dim()
        s = [np.newaxis]*self.dimensions
        w0 = output.copy()
        output.fill(0)
        for key, val in self.stencil_matrix(self.shape(False)).items():
            M = self.N if key >= 0 else self.dim()
            s1 = slice(max(0, -key), M-max(0, key))
            Q = s1.stop-s1.start
            s2 = self.sl[slice(max(0, key), Q+max(0, key))]
            sk = self.sl[s1]
            s[self.axis] = slice(0, Q)
            output[sk] += val[tuple(s)]*w0[s2]
        output[self.sl[slice(-nbcs, None)]] = 0

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

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)

        if i < self.dim():
            basis = np.zeros(self.shape(True))
            M = self.stencil_matrix()
            row = M.diags().getrow(i)
            indices = row.indices
            vals = row.data
            basis[indices] = vals
            basis = n_cheb.Chebyshev(basis)
            if k > 0:
                basis = basis.deriv(k)
            output_array[:] = basis(x)
        else:
            output_array[:] = self.get_bc_basis().evaluate_basis_derivative(x, i-self.dim(), k)
        return output_array

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w = self.to_ortho(u)
        output_array[:] = chebval(x, w)
        return output_array

    def get_refined(self, N):
        return self.__class__(N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc,
                              scaled=self._scaled)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return self.__class__(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=padding_factor,
                              dealias_direct=dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc,
                              scaled=self._scaled)

    def get_unplanned(self):
        return self.__class__(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc,
                              scaled=self._scaled)

class CompositeSpaceU(OrthogonalU):
    """Common class for all spaces based on composite bases of Chebyshev
    polynomials of second kind"""

    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1., 1.), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        OrthogonalU.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                             padding_factor=padding_factor, dealias_direct=dealias_direct,
                             coordinates=coordinates)
        ChebyshevBase.plan(self, (int(padding_factor*N),), 0, dtype, {})
        from shenfun.tensorproductspace import BoundaryValues
        self._scaled = scaled
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    def plan(self, shape, axis, dtype, options):
        OrthogonalU.plan(self, shape, axis, dtype, options)
        ChebyshevBase.plan(self, shape, axis, dtype, options)

    def evaluate_basis_all(self, x=None, argument=0):
        V = OrthogonalU.evaluate_basis_all(self, x=x, argument=argument)
        return self._composite(V, argument=argument)

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        V = OrthogonalU.evaluate_basis_derivative_all(self, x=x, k=k, argument=argument)
        return self._composite(V, argument=argument)

    def _evaluate_expansion_all(self, input_array, output_array, x=None, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_expansion_all(self, input_array, output_array, x, False)
            return
        assert input_array is self.backward.tmp_array
        assert output_array is self.backward.output_array
        input_array[:] = self.to_ortho(input_array, output_array)
        OrthogonalU._evaluate_expansion_all(self, input_array, output_array, x, True)

    @property
    def is_orthogonal(self):
        return False

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
        raise NotImplementedError

    def get_refined(self, N):
        return self.__class__(N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc,
                              scaled=self._scaled)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return self.__class__(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=padding_factor,
                              dealias_direct=dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc,
                              scaled=self._scaled)

    def get_unplanned(self):
        return self.__class__(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc,
                              scaled=self._scaled)

class ShenDirichlet(CompositeSpace):
    r"""Function space for Dirichlet boundary conditions

    The basis function is

    .. math::

        \phi_k = T_k - T_{k+2}

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

    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1., 1.), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'SD'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = np.ones(N, dtype=int)
        d[-2:] = 0
        return SparseMatrix({0: d, 2: -d[:-2]}, (N, N))

    def slice(self):
        return slice(0, self.N-2)

    def _evaluate_scalar_product(self, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            self.scalar_product.output_array[self.si[-2]] = 0
            self.scalar_product.output_array[self.si[-1]] = 0
            return
        Orthogonal._evaluate_scalar_product(self, True)
        output = self.scalar_product.output_array
        s0 = self.sl[slice(0, self.N-2)]
        s1 = self.sl[slice(2, self.N)]
        output[s0] -= output[s1]
        output[self.si[-2]] = 0
        output[self.si[-1]] = 0

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)
        s0 = self.sl[slice(0, self.N-2)]
        s1 = self.sl[slice(2, self.N)]
        output_array[s0] = input_array[s0]
        output_array[s1] -= input_array[s0]
        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCDirichlet(self.N, quad=self.quad, domain=self.domain,
                                     coordinates=self.coors.coordinates)
        return self._bc_basis

class DirichletU(CompositeSpaceU):
    r"""Function space for Dirichlet boundary conditions

    The basis function is

    .. math::

        \phi_k = U_k - (k+1)/(k+3)U_{k+2}

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

    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1., 1.), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpaceU.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                                 padding_factor=padding_factor, dealias_direct=dealias_direct,
                                 coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self._scaled = scaled
        self._factor = np.ones(1)
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'DU'

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def sympy_basis(self, i=0, x=xp):
        assert i < self.N
        if i < self.N-2:
            return sp.chebyshevu(i, x) - sp.chebyshevu(i+2, x)*(i+1)/(i+3)
        if i == self.N-2:
            return 0.5*(1-x)
        return 0.5*(1+x)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i < self.N-2:
            w = np.arccos(x)
            output_array[:] = np.sin((i+1)*w)/np.sin(w) - (i+1)/(i+3)*np.sin((i+3)*w)/np.sin(w)
        elif i == self.N-2:
            output_array[:] = 0.5*(1-x)
        elif i == self.N-1:
            output_array[:] = 0.5*(1+x)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        x = np.atleast_1d(x)
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)

        if i < self.N-2:
            basis = np.zeros(self.shape(True)+1)
            basis[i+1] = 1/(i+1)
            basis[i+3] = -(i+1)/(i+3)**2
            basis = n_cheb.Chebyshev(basis)
            k += 1
            basis = basis.deriv(k)
            output_array[:] = basis(x)

        elif i == self.N-2:
            output_array[:] = 0
            if k == 1:
                output_array[:] = -0.5
            elif k == 0:
                output_array[:] = 0.5*(1-x)
        elif i == self.N-1:
            output_array[:] = 0
            if k == 1:
                output_array[:] = 0.5
            elif k == 0:
                output_array[:] = 0.5*(1+x)

        return output_array

    def set_factor_array(self, v):
        """Set intermediate factor arrays"""
        if not self._factor.shape == v.shape:
            k = np.arange(self.N)
            self._factor = self.broadcast_to_ndims((k+1)/(k+3))

    def _composite(self, V, argument=0):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1])
        P[:, :-2] = V[:, :-2] - V[:, 2:]*(k[1:-1]/(k[:-2]+3))
        if argument == 1: # if trial function
            P[:, -1] = (V[:, 0] + V[:, 1])/2    # x = +1
            P[:, -2] = (V[:, 0] - V[:, 1])/2    # x = -1
        return P

    def _evaluate_scalar_product(self, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            self.scalar_product.output_array[self.si[-2]] = 0
            self.scalar_product.output_array[self.si[-1]] = 0
            return
        OrthogonalU._evaluate_scalar_product(self, True)
        output = self.scalar_product.output_array
        s0 = self.sl[slice(0, self.N-2)]
        s1 = self.sl[slice(2, self.N)]
        #w0 = output.copy()
        self.set_factor_array(output)
        output[s0] -= self._factor[s0]*output[s1]
        output[self.si[-2]] = 0
        output[self.si[-1]] = 0

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)
        s0 = self.sl[slice(0, self.N-2)]
        s1 = self.sl[slice(2, self.N)]
        self.set_factor_array(output_array)
        output_array[s0] = input_array[s0]
        output_array[s1] -= input_array[s0]*self._factor[s0]
        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        fj = self.evaluate_basis_all(x)
        output_array[:] = np.dot(fj, u)
        return output_array

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCDirichlet(self.N, quad=self.quad, domain=self.domain,
                                     coordinates=self.coors.coordinates)
        return self._bc_basis

class Heinrichs(CompositeSpace):
    r"""Function space for Dirichlet boundary conditions

    The basis function is

    .. math::

        \phi_k = (1-x^2)T_k

    If scaled is True it is

    .. math::

        \phi_k = (1-x^2)T_k/(k+1)/(k+2)

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
        dtype : data-type, optiona
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

    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1., 1.), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'HH'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = 0.5*np.ones(N, dtype=int)
        d[-2:] = 0
        d[1] = 0.25
        dm2 = -0.25*np.ones(N-2, dtype=int)
        dm2[-2:] = 0
        dp2 = -0.25*np.ones(N-2, dtype=int)
        dp2[0] = -0.5
        if self.is_scaled():
            k = np.arange(N)
            d /= ((k+1)*(k+2))
            dm2 /= ((k[:-2]+3)*(k[:-2]+4))
            dp2 /= ((k[:-2]+1)*(k[:-2]+2))
        return SparseMatrix({-2: dm2, 0: d, 2: dp2}, (N, N))

    def slice(self):
        return slice(0, self.N-2)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCDirichlet(self.N, quad=self.quad, domain=self.domain,
                                     coordinates=self.coors.coordinates)
        return self._bc_basis

class QuasiDirichlet(CompositeSpace):
    r"""Function space for Dirichlet boundary conditions

    The basis function is

    .. math::

        \phi_k = \frac{2(1-x^2)^2 T_{k+2}''}{\pi (k+3)(k+2)^2(k+1)}

    Note
    ----
    Using this basis as testfunction corresponds to using the quasi-inverse
    approach. The basis actually has 4 boundary conditions for all k<N-6,
    u(+-1)=0 and u'(+-1)=0, but it is used here as a basis of dimension N-2,
    by truncating for the highest wavenumbers.

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

    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1., 1.), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'MD'

    @staticmethod
    def scale(k, sympy=False):
        pi = sp.pi if sympy else np.pi
        return 2/(pi*(k+3)*(k+2)**2*(k+1))

    def factor(self):
        k = self.wavenumbers()
        self._factor = 2/(np.pi*(k+3)*(k+2)**2*(k+1))
        return self._factor

    def sympy_basis(self, i=0, x=xp):
        assert i < self.N
        if i < self.N-2:
            return self.scale(i, True)*(1-x**2)**2*sp.chebyshevt(i+2, x).diff(x, 2)
        if i == self.N-2:
            return 0.5*(1-x)
        return 0.5*(1+x)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i < self.N-2:
            w = np.arccos(x)
            output_array[:] = -self.scale(i, False)*np.sin(w)**2*(i+2)*((i+3)*np.cos((i+2)*w)-np.sin((i+3)*w)/np.sin(w))
        elif i == self.N-2:
            output_array[:] = 0.5*(1-x)
        elif i == self.N-1:
            output_array[:] = 0.5*(1+x)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = self.sympy_basis(i)
        xp = basis.free_symbols.pop()
        output_array[:] = sp.lambdify(xp, basis.diff(xp, k))(x)
        return output_array

    def is_scaled(self):
        """Return True if scaled basis is used, otherwise False"""
        return self._scaled

    def _composite(self, V, argument=0):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1])
        P[:, :-2] = V[:, :-2]/(2*np.pi*k[2:]*k[1:-1])
        #P[:, :-4] -= V[:, 2:-2]*(1/(np.pi*k[3:-1]*k[1:-3]))
        #P[:, :-6] += V[:, 4:-2]/(2*np.pi*k[2:-4]*k[3:-3])
        P[:, :-2] -= V[:, 2:]*(1/(np.pi*(k[:-2]+3)*(k[:-2]+1)))
        P[:, :-4] += V[:, 4:]/(2*np.pi*k[2:-2]*k[3:-1])


        if argument == 1: # if trial function
            P[:, -1] = (V[:, 0] + V[:, 1])/2    # x = +1
            P[:, -2] = (V[:, 0] - V[:, 1])/2    # x = -1
        return P

    def _evaluate_scalar_product(self, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            self.scalar_product.output_array[self.si[-2]] = 0
            self.scalar_product.output_array[self.si[-1]] = 0
            return
        Orthogonal._evaluate_scalar_product(self, True)
        output = self.scalar_product.output_array
        wk = output.copy()
        k = self.broadcast_to_ndims(np.arange(self.N))
        s0 = self.sl[slice(0, self.N-2)]
        output[s0] = wk[s0]/(2*np.pi*(k[s0]+1)*(k[s0]+2))
        #s0 = self.sl[slice(0, self.N-4)]
        #s2 = self.sl[slice(2, self.N-2)]
        s0 = self.sl[slice(0, self.N-2)]
        s2 = self.sl[slice(2, self.N)]
        output[s0] -= wk[s2]*(1/((k[s0]+1)*(k[s0]+3))/np.pi)
        #s0 = self.sl[slice(0, self.N-6)]
        #s4 = self.sl[slice(4, self.N-2)]
        s0 = self.sl[slice(0, self.N-4)]
        s4 = self.sl[slice(4, self.N)]
        output[s0] += wk[s4]/(2*np.pi*(k[s0]+2)*(k[s0]+3))

        output[self.si[-2]] = 0
        output[self.si[-1]] = 0

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)

        k = self.broadcast_to_ndims(np.arange(self.N))
        s0 = self.sl[slice(0, self.N-2)]
        output_array[s] = input_array[s0]/(2*np.pi*(k[s0]+1)*(k[s0]+2))
        #s0 = self.sl[slice(0, self.N-4)]
        #s2 = self.sl[slice(2, self.N-2)]
        s0 = self.sl[slice(0, self.N-2)]
        s2 = self.sl[slice(2, self.N)]
        output_array[s2] -= input_array[s0]*(1/(np.pi*(k[s0]+1)*(k[s0]+3)))
        #s0 = self.sl[slice(0, self.N-6)]
        #s4 = self.sl[slice(4, self.N-2)]
        s0 = self.sl[slice(0, self.N-4)]
        s4 = self.sl[slice(4, self.N)]
        output_array[s4] += input_array[s0]/(2*np.pi*(k[s0]+2)*(k[s0]+3))

        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w = self.to_ortho(u)
        output_array[:] = chebval(x, w)
        return output_array

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCDirichlet(self.N, quad=self.quad, domain=self.domain,
                                     coordinates=self.coors.coordinates)
        return self._bc_basis

class ShenNeumann(CompositeSpace):
    """Function space for homogeneous Neumann boundary conditions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

        mean : float, optional
            Mean value if solving a Poisson problem that only is known
            up to a constant. If mean is None, then we solve also for
            basis function 0, which is T_0. A Helmholtz problem can use
            mean=None.
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
    """
    def __init__(self, N, quad="GC", mean=None, bc=(0, 0), domain=(-1., 1.), dtype=float, padding_factor=1,
                 scaled=False, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)
        self.mean = mean

    @staticmethod
    def boundary_condition():
        return 'Neumann'

    @staticmethod
    def short_name():
        return 'SN'

    @property
    def use_fixed_gauge(self):
        if self.mean is None:
            return False
        T = self.tensorproductspace
        if T:
            return T.use_fixed_gauge
        return True

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = np.ones(N, dtype=int)
        d[-2:] = 0
        k = np.arange(N-2)
        return SparseMatrix({0: d, 2: -(k/(k+2))**2}, (N, N))

    def slice(self):
        return slice(0, self.N-2)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCNeumann(self.N, quad=self.quad, domain=self.domain,
                                   coordinates=self.coors.coordinates)
        return self._bc_basis

    def get_refined(self, N):
        return ShenNeumann(N,
                           quad=self.quad,
                           bc=self.bc.bc,
                           domain=self.domain,
                           dtype=self.dtype,
                           scaled=self._scaled,
                           padding_factor=self.padding_factor,
                           dealias_direct=self.dealias_direct,
                           coordinates=self.coors.coordinates,
                           mean=self.mean)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return ShenNeumann(self.N,
                           quad=self.quad,
                           bc=self.bc.bc,
                           domain=self.domain,
                           dtype=self.dtype,
                           scaled=self._scaled,
                           padding_factor=padding_factor,
                           dealias_direct=dealias_direct,
                           coordinates=self.coors.coordinates,
                           mean=self.mean)

    def get_unplanned(self):
        return ShenNeumann(self.N,
                           quad=self.quad,
                           bc=self.bc.bc,
                           domain=self.domain,
                           dtype=self.dtype,
                           scaled=self._scaled,
                           padding_factor=self.padding_factor,
                           dealias_direct=self.dealias_direct,
                           coordinates=self.coors.coordinates,
                           mean=self.mean)

class CombinedShenNeumann(CompositeSpace):
    """Function space for homogeneous Neumann boundary conditions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

        mean : float, optional
            Mean value if solving a Poisson problem that only is known
            up to a constant. If mean is None, then we solve also for
            basis function 0, which is T_0. A Helmholtz problem can use
            mean=None.
        bc : 2-tuple of floats, optional
            Boundary conditions at, respectively, x=(-1, 1).
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
    def __init__(self, N, quad="GC", mean=None, bc=(0, 0), domain=(-1., 1.), dtype=float, padding_factor=1,
                 scaled=False, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)
        self.mean = mean

    @staticmethod
    def boundary_condition():
        return 'Neumann'

    @staticmethod
    def short_name():
        return 'CN'

    @property
    def use_fixed_gauge(self):
        if self.mean is None:
            return False
        T = self.tensorproductspace
        if T:
            return T.use_fixed_gauge
        return True

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        k = np.arange(N)
        k[0] = 1
        d = 2/k**2
        d[-2:] = 0
        d[0] = 1
        d[1] = 1
        d[2] = 0.25
        dm2 = -1/k[:-2]**2
        dm2[0] = 0
        dm2[-2:] = 0
        dp2 = -1/k[2:]**2
        dp2[0] = 0
        return SparseMatrix({-2: dm2, 0: d, 2: dp2}, (N, N))

    def slice(self):
        return slice(0, self.N-2)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCNeumann(self.N, quad=self.quad, domain=self.domain,
                                   coordinates=self.coors.coordinates)
        return self._bc_basis

    def get_refined(self, N):
        return CombinedShenNeumann(N,
                           quad=self.quad,
                           bc=self.bc.bc,
                           scaled=self._scaled,
                           domain=self.domain,
                           dtype=self.dtype,
                           padding_factor=self.padding_factor,
                           dealias_direct=self.dealias_direct,
                           coordinates=self.coors.coordinates,
                           mean=self.mean)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return CombinedShenNeumann(self.N,
                           quad=self.quad,
                           bc=self.bc.bc,
                           scaled=self._scaled,
                           domain=self.domain,
                           dtype=self.dtype,
                           padding_factor=padding_factor,
                           dealias_direct=dealias_direct,
                           coordinates=self.coors.coordinates,
                           mean=self.mean)

    def get_unplanned(self):
        return CombinedShenNeumann(self.N,
                           quad=self.quad,
                           bc=self.bc.bc,
                           scaled=self._scaled,
                           domain=self.domain,
                           dtype=self.dtype,
                           padding_factor=self.padding_factor,
                           dealias_direct=self.dealias_direct,
                           coordinates=self.coors.coordinates,
                           mean=self.mean)

class MikNeumann(CompositeSpace):
    r"""Function space for homogeneous Neumann boundary conditions

    The basis function is

    .. math::

        \phi_k = \frac{2}{k+1}\int (T_{k-1}-T_{k+1})

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

        mean : float, optional
            Mean value if solving a Poisson problem that only is known
            up to a constant. If mean is None, then we solve also for
            basis function 0, which is T_0. A Helmholtz problem can use
            mean=None.
        bc : 2-tuple of floats, optional
            Boundary conditions at, respectively, x=(-1, 1).
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
    def __init__(self, N, quad="GC", mean=None, bc=(0, 0), domain=(-1., 1.), dtype=float, padding_factor=1,
                 scaled=False, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)
        self.mean = mean

    @staticmethod
    def boundary_condition():
        return 'Neumann'

    @staticmethod
    def short_name():
        return 'MN'

    @property
    def use_fixed_gauge(self):
        if self.mean is None:
            return False
        T = self.tensorproductspace
        if T:
            return T.use_fixed_gauge
        return True

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        k = np.arange(N)
        k[0] = 1
        d = 2/k/(k+1)
        d[-2:] = 0
        d[0] = 1
        d[1] = 3/2
        d[2] = 1/3
        dm2 = -1/k[:-2]/(k[2:]+1)
        dm2[0] = 0
        dm2[-2:] = 0
        dp2 = -1/k[2:]/(k[2:]-1)
        dp2[0] = 0
        dp2[1] = -1/6
        dp2[2] = -1/12
        return SparseMatrix({-2: dm2, 0: d, 2: dp2}, (N, N))

    def slice(self):
        return slice(0, self.N-2)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCNeumann(self.N, quad=self.quad, domain=self.domain,
                                   coordinates=self.coors.coordinates)
        return self._bc_basis

    def get_refined(self, N):
        return MikNeumann(N,
                           quad=self.quad,
                           bc=self.bc.bc,
                           domain=self.domain,
                           dtype=self.dtype,
                           padding_factor=self.padding_factor,
                           dealias_direct=self.dealias_direct,
                           coordinates=self.coors.coordinates,
                           scaled=self._scaled,
                           mean=self.mean)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return MikNeumann(self.N,
                           quad=self.quad,
                           bc=self.bc.bc,
                           domain=self.domain,
                           dtype=self.dtype,
                           padding_factor=padding_factor,
                           dealias_direct=dealias_direct,
                           coordinates=self.coors.coordinates,
                           scaled=self._scaled,
                           mean=self.mean)

    def get_unplanned(self):
        return MikNeumann(self.N,
                           quad=self.quad,
                           bc=self.bc.bc,
                           domain=self.domain,
                           dtype=self.dtype,
                           padding_factor=self.padding_factor,
                           dealias_direct=self.dealias_direct,
                           coordinates=self.coors.coordinates,
                           scaled=self._scaled,
                           mean=self.mean)

class ShenBiharmonic(CompositeSpace):
    """Function space for biharmonic equation

    Using 2 Dirichlet and 2 Neumann boundary conditions. All possibly
    nonhomogeneous.

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss
        bc : 4-tuple of numbers
            The values of the 4 boundary conditions at x=(-1, 1).
            The two Dirichlet at (-1, 1) first and then the Neumann at (-1, 1).
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
    def __init__(self, N, quad="GC", bc=(0, 0, 0, 0), domain=(-1., 1.), dtype=float,
                 scaled=False, padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, scaled=scaled, bc=bc,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    @staticmethod
    def short_name():
        return 'SB'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = np.ones(N, dtype=int)
        d[-4:] = 0
        k = np.arange(N)
        d2 = - (2*(k[:-2]+2)/(k[:-2]+3))
        d2[-2:] = 0
        d4 = (k[:-4]+1)/(k[:-4]+3)
        return SparseMatrix({0: d, 2: d2, 4: d4}, (N, N))

    def slice(self):
        return slice(0, self.N-4)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCBiharmonic(self.N, quad=self.quad, domain=self.domain,
                                      coordinates=self.coors.coordinates)
        return self._bc_basis

class SecondNeumann(CompositeSpace): #pragma: no cover
    """Function space for homogeneous second order Neumann boundary conditions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

        mean : float, optional
            Mean value
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

    def __init__(self, N, quad="GC", mean=0, domain=(-1., 1.), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)
        self.mean = mean
        self._factor = np.zeros(0)

    @staticmethod
    def boundary_condition():
        return 'Neumann2'

    @staticmethod
    def short_name():
        return 'TN'

    def _composite(self, V, argument=0):
        assert self.N == V.shape[1]
        P = np.zeros_like(V)
        k = np.arange(self.N).astype(float)
        P[:, :-2] = V[:, :-2] - (k[:-2]/(k[:-2]+2))**2*(k[:-2]**2-1)/((k[:-2]+2)**2-1)*V[:, 2:]
        return P

    def sympy_basis(self, i=0, x=xp):
        assert i < self.N-2
        return sp.chebyshevt(i, x) - (i/(i+2))**2*(i**2-1)/((i+2)**2-1)*sp.chebyshevt(i+2, x)

    def evaluate_basis(self, x, i=0, output_array=None):
        assert i < self.N-2
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        w = np.arccos(x)
        output_array[:] = np.cos(i*w) - (i*1./(i+2))**2*(i**2-1.)/((i+2)**2-1.)*np.cos((i+2)*w)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        assert i < self.N-2
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[np.array([i, i+2])] = (1, -(i*1./(i+2))**2*(i**2-1.)/((i+2)**2-1.))
        basis = n_cheb.Chebyshev(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def set_factor_array(self, v):
        """Set intermediate factor arrays"""
        if not self._factor.shape == v.shape:
            k = self.wavenumbers().astype(float)
            self._factor = (k/(k+2))**2*(k**2-1)/((k+2)**2-1)

    def _evaluate_scalar_product(self, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            self.scalar_product.output_array[self.sl[slice(-2, None)]] = 0
            return
        Orthogonal._evaluate_scalar_product(self, True)
        output = self.scalar_product.output_array
        self.set_factor_array(output)
        sm2 = self.sl[slice(0, -2)]
        s2p = self.sl[slice(2, None)]
        output[sm2] -= self._factor*output[s2p]
        output[self.sl[slice(-2, None)]] = 0

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(2, None)]
        self.set_factor_array(input_array)
        output_array[s0] = input_array[s0]
        output_array[s1] -= self._factor*input_array[s0]
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_array(u)
        output_array[:] = chebval(x, u[:-2])
        w_hat[2:] = self._factor*u[:-2]
        output_array -= chebval(x, w_hat)
        return output_array

    def get_refined(self, N):
        return SecondNeumann(N,
                             quad=self.quad,
                             domain=self.domain,
                             dtype=self.dtype,
                             padding_factor=self.padding_factor,
                             dealias_direct=self.dealias_direct,
                             coordinates=self.coors.coordinates,
                             mean=self.mean)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return SecondNeumann(self.N,
                             quad=self.quad,
                             domain=self.domain,
                             dtype=self.dtype,
                             padding_factor=padding_factor,
                             dealias_direct=dealias_direct,
                             coordinates=self.coors.coordinates,
                             mean=self.mean)

    def get_unplanned(self):
        return SecondNeumann(self.N,
                             quad=self.quad,
                             domain=self.domain,
                             dtype=self.dtype,
                             padding_factor=self.padding_factor,
                             dealias_direct=self.dealias_direct,
                             coordinates=self.coors.coordinates,
                             mean=self.mean)

class UpperDirichlet(CompositeSpace):
    """Function space with homogeneous Dirichlet on upper edge (x=1) of boundary

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

    """

    def __init__(self, N, quad="GC", bc=(None, 0), domain=(-1., 1.), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'UpperDirichlet'

    @staticmethod
    def short_name():
        return 'UD'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        d = np.ones(N, dtype=int)
        d[-1:] = 0
        return SparseMatrix({0: d, 1: -d[:-1]}, (N, N))

    def slice(self):
        return slice(0, self.N-1)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCUpperDirichlet(self.N, quad=self.quad, domain=self.domain,
                                          coordinates=self.coors.coordinates)
        return self._bc_basis

class ShenBiPolar(Orthogonal):
    """Function space for the Biharmonic equation in polar coordinates

    Parameters
    ----------
        N : int
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

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
    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1., 1.), dtype=float,
                 scaled=False, padding_factor=1, dealias_direct=False, coordinates=None):
        self.forward = functools.partial(self.forward, fast_transform=False)
        self.backward = functools.partial(self.backward, fast_transform=False)
        self.scalar_product = functools.partial(self.scalar_product, fast_transform=False)
        Orthogonal.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                            padding_factor=padding_factor, dealias_direct=dealias_direct,
                            coordinates=coordinates)

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
        if self.padding_factor != 1:
            trunc_array = self._get_truncarray(shape, V.dtype)
            self.forward = Transform(self.forward, None, U, V, trunc_array)
            self.backward = Transform(self.backward, None, trunc_array, V, U)
        else:
            self.forward = Transform(self.forward, None, U, V, V)
            self.backward = Transform(self.backward, None, V, V, U)
        self.scalar_product = Transform(self.scalar_product, None, U, V, V)
        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    @staticmethod
    def short_name():
        return 'SP'

    @property
    def is_orthogonal(self):
        return False

    @property
    def has_nonhomogeneous_bcs(self):
        return False

    def to_ortho(self, input_array, output_array=None):
        raise(NotImplementedError)

    def slice(self):
        return slice(0, self.N-4)

    def sympy_basis(self, i=0, x=xp):
        f = (1-x)**2*(1+x)**2*(sp.chebyshevt(i+1, x).diff(x, 1))
        return f

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        f = self.sympy_basis(i, xp)
        output_array[:] = sp.lambdify(xp, f)(x)
        return output_array

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            x = self.mesh(False, False)
        output_array = np.zeros((x.shape[0], self.N))
        D = np.zeros(x.shape[0])
        for j in range(self.N-4):
            output_array[:, j] = self.evaluate_basis(x, j, D)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        f = self.sympy_basis(i, xp).diff(xp, k)
        output_array[:] = sp.lambdify(xp, f)(x)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mesh(False, False)
        output_array = np.zeros((x.shape[0], self.N))
        D = np.zeros(x.shape[0])
        for j in range(self.N-4):
            output_array[:, j] = self.evaluate_basis_derivative(x, j, k, output_array=D)
        return output_array

    def _evaluate_scalar_product(self, fast_transform=False):
        SpectralBase._evaluate_scalar_product(self)
        self.scalar_product.output_array[self.sl[slice(-4, None)]] = 0

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        fj = self.evaluate_basis_all(x)
        output_array[:] = np.dot(fj, u)
        return output_array

class DirichletNeumann(CompositeSpace):
    """Function space for mixed Dirichlet/Neumann boundary conditions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

        bc : tuple of numbers
            Boundary conditions at edges of domain
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

    """

    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1., 1.), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct, coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'DirichletNeumann'

    @staticmethod
    def short_name():
        return 'DN'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        k = np.arange(N)
        d = np.ones(N)
        d[-2:] = 0
        d1 = ((-k[:-1]**2 + (k[:-1]+2)**2)/((k[:-1]+1)**2 + (k[:-1]+2)**2))
        d2 = ((-k[:-2]**2 - (k[:-2]+1)**2)/((k[:-2]+1)**2 + (k[:-2]+2)**2))
        return SparseMatrix({0: d, 1: d1, 2: d2}, (N, N))

    def slice(self):
        return slice(0, self.N-2)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCDirichletNeumann(self.N, quad=self.quad, domain=self.domain,
                                            coordinates=self.coors.coordinates)
        return self._bc_basis

class NeumannDirichlet(CompositeSpace):
    """Function space for mixed Neumann/Dirichlet boundary conditions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

        bc : tuple of numbers
            Boundary conditions at edges of domain
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

    """

    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1., 1.), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct, coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'NeumannDirichlet'

    @staticmethod
    def short_name():
        return 'ND'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        k = np.arange(N)
        d = np.ones(N)
        d[-2:] = 0
        d1 = (-(-k[:-1]**2 + (k[:-1]+2)**2)/((k[:-1]+1)**2 + (k[:-1]+2)**2))
        d2 = ((-k[:-2]**2 - (k[:-2]+1)**2)/((k[:-2]+1)**2 + (k[:-2]+2)**2))
        return SparseMatrix({0: d, 1: d1, 2: d2}, (N, N))

    def slice(self):
        return slice(0, self.N-2)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCNeumannDirichlet(self.N, quad=self.quad, domain=self.domain,
                                            coordinates=self.coors.coordinates)
        return self._bc_basis

class UpperDirichletNeumann(CompositeSpace):
    """Function space for both Dirichlet and Neumann boundary conditions
    on the right hand side

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
            Type of quadrature

            - GL - Chebyshev-Gauss-Lobatto
            - GC - Chebyshev-Gauss

        bc : tuple of numbers
            Boundary conditions at the right edge of domain
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

    """

    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1., 1.), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype, bc=bc, scaled=scaled,
                                padding_factor=padding_factor, dealias_direct=dealias_direct, coordinates=coordinates)

    @staticmethod
    def boundary_condition():
        return 'UpperDirichletNeumann'

    @staticmethod
    def short_name():
        return 'US'

    def stencil_matrix(self, N=None):
        N = self.N if N is None else N
        k = np.arange(N)
        d = np.ones(N)
        d[-2:] = 0
        d1 = (-4*(k[:-1]+1)/(2*k[:-1]+3))
        d2 = ((2*k[:-2]+1)/(2*k[:-2]+3))
        return SparseMatrix({0: d, 1: d1, 2: d2}, (N, N))

    def slice(self):
        return slice(0, self.N-2)

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCUpperDirichletNeumann(self.N, quad=self.quad, domain=self.domain,
                                                 coordinates=self.coors.coordinates)
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

    def __init__(self, N, quad="GC", domain=(-1., 1.), scaled=False,
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
        return n_cheb.chebvander(x, self.num_T-1)

    def _composite(self, V, argument=1):
        N = self.shape()
        P = np.zeros(V[:, :N].shape)
        P[:] = np.tensordot(V[:, :self.num_T], self.stencil_matrix(), (1, 1))
        return P

    def sympy_basis(self, i=0, x=xp):
        M = self.stencil_matrix()
        return np.sum(M[i]*np.array([sp.chebyshevt(j, x) for j in range(self.num_T)]))

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

class BCDirichlet(BCBase):

    @staticmethod
    def short_name():
        return 'BCD'

    def stencil_matrix(self, N=None):
        return sp.Rational(1/2)*np.array([[1, -1],
                                          [1, 1]])

class BCNeumann(BCBase):

    @staticmethod
    def short_name():
        return 'BCN'

    def stencil_matrix(self, N=None):
        return np.array([[0, 1/2, -1/8],
                         [0, 1/2, 1/8]])

class BCBiharmonic(BCBase):

    @staticmethod
    def short_name():
        return 'BCB'

    def stencil_matrix(self, N=None):
        return np.array([[0.5, -9./16., 0, 1./16],
                         [0.5, 9./16., 0, -1./16.],
                         [1./8., -1./16., -1./8., 1./16.],
                         [-1./8., -1./16., 1./8., 1./16.]])

class BCUpperDirichlet(BCBase):

    @staticmethod
    def short_name():
        return 'BCUD'

    def stencil_matrix(self, N=None):
        return np.array([[0.5, 0.5]])

class BCNeumannDirichlet(BCBase):

    @staticmethod
    def short_name():
        return 'BCND'

    def stencil_matrix(self, N=None):
        return np.array([[1, -0.6, -0.4],
                         [1, 0, 0]])

class BCDirichletNeumann(BCBase):

    @staticmethod
    def short_name():
        return 'BCDN'

    def stencil_matrix(self, N=None):
        return np.array([[1, 0],
                         [1, 1]])

class BCUpperDirichletNeumann(BCBase):

    @staticmethod
    def short_name():
        return 'BCUDN'

    def stencil_matrix(self, N=None):
        return np.array([[1, 0, 0],
                         [1, -5/3, 2/3]])

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
