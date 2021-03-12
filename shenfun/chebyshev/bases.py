"""
Module for defining function spaces in the Chebyshev family
"""
from __future__ import division
import functools
import numpy as np
import sympy as sp
from numpy.polynomial import chebyshev as n_cheb
from scipy.special import eval_chebyt, eval_chebyu
from mpi4py_fft import fftw
from shenfun.spectralbase import SpectralBase, work, Transform, FuncWrap, \
    islicedict, slicedict
from shenfun.optimization import optimizer

__all__ = ['ChebyshevBase', 'Orthogonal', 'ShenDirichlet',
           'ShenNeumann', 'ShenBiharmonic',
           'SecondNeumann', 'UpperDirichlet',
           'UpperDirichletNeumann', 'Heinrichs',
           'ShenBiPolar', 'BCDirichlet', 'BCBiharmonic',
           'DirichletNeumann',
           'CombinedShenNeumann', 'MikNeumann',
           'DirichletU']


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
                weights = np.pi/(N+1)

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

        if 'builders' in self._xfftn_fwd.func.__module__: #pragma: no cover
            opts = dict(
                avoid_copy=True,
                overwrite_input=True,
                auto_align_input=True,
                auto_contiguous=True,
                planner_effort='FFTW_MEASURE',
                threads=1,
            )
            opts.update(options)

            U = fftw.aligned(shape, dtype=float)
            xfftn_fwd = plan_fwd(U, axis=axis, **opts)
            V = xfftn_fwd.output_array
            xfftn_bck = plan_bck(V, axis=axis, **opts)
            V.fill(0)
            U.fill(0)

            xfftn_fwd.update_arrays(U, V)
            xfftn_bck.update_arrays(V, U)
        else: # fftw wrapped with mpi4py-fft
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
        if quad == 'GC':
            self._xfftn_fwd = functools.partial(fftw.dstn, type=2)
            self._xfftn_bck = functools.partial(fftw.dstn, type=3)

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
        basis = np.zeros(self.shape(True))
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
            self.scalar_product._input_array /= self.broadcast_to_ndims(np.sin((np.arange(self.N)+0.5)*np.pi/self.N))
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
            s0 = self.si[self.N-1]
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

        if 'builders' in self._xfftn_fwd.func.__module__: #pragma: no cover
            opts = dict(
                avoid_copy=True,
                overwrite_input=True,
                auto_align_input=True,
                auto_contiguous=True,
                planner_effort='FFTW_MEASURE',
                threads=1,
            )
            opts.update(options)

            U = fftw.aligned(shape, dtype=float)
            xfftn_fwd = plan_fwd(U, axis=axis, **opts)
            V = xfftn_fwd.output_array
            xfftn_bck = plan_bck(V, axis=axis, **opts)
            V.fill(0)
            U.fill(0)

            xfftn_fwd.update_arrays(U, V)
            xfftn_bck.update_arrays(V, U)
        else: # fftw wrapped with mpi4py-fft
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

    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1., 1.), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        Orthogonal.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                            padding_factor=padding_factor, dealias_direct=dealias_direct,
                            coordinates=coordinates)
        ChebyshevBase.plan(self, (int(padding_factor*N),), 0, dtype, {})

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

    @property
    def is_orthogonal(self):
        return False

    def _composite(self, V, argument=0):
        """Return Vandermonde matrix V adjusted for basis composition

        Parameters
        ----------
        V : Vandermonde type matrix
        argument : int
                Zero for test and 1 for trialfunction

        """
        raise NotImplementedError

class CompositeSpaceU(OrthogonalU):
    """Common class for all spaces based on composite bases of Chebyshev
    polynomials of second kind"""

    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1., 1.), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        OrthogonalU.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                             padding_factor=padding_factor, dealias_direct=dealias_direct,
                             coordinates=coordinates)
        ChebyshevBase.plan(self, (int(padding_factor*N),), 0, dtype, {})

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

    def _composite(self, V, argument=0):
        """Return Vandermonde matrix V adjusted for basis composition

        Parameters
        ----------
        V : Vandermonde type matrix
        argument : int
                Zero for test and 1 for trialfunction

        """
        raise NotImplementedError


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
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
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
        return 'SD'

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def sympy_basis(self, i=0, x=xp):
        assert i < self.N
        if i < self.N-2:
            return sp.chebyshevt(i, x) - sp.chebyshevt(i+2, x)
        if i == self.N-2:
            return 0.5*(1-x)
        return 0.5*(1+x)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i < self.N-2:
            w = np.arccos(x)
            output_array[:] = np.cos(i*w) - np.cos((i+2)*w)
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
        if i < self.N-2:
            basis = np.zeros(self.shape(True))
            basis[np.array([i, i+2])] = (1, -1)
            basis = n_cheb.Chebyshev(basis)
            if k > 0:
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

    def is_scaled(self):
        """Return True if scaled basis is used, otherwise False"""
        return False

    def _composite(self, V, argument=0):
        P = np.zeros_like(V)
        P[:, :-2] = V[:, :-2] - V[:, 2:]
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

    def slice(self):
        return slice(0, self.N-2)

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        output_array[:] = n_cheb.chebval(x, u[:-2])
        w_hat[2:] = u[:-2]
        output_array -= n_cheb.chebval(x, w_hat)
        output_array += 0.5*(u[-1]*(1+x)+u[-2]*(1-x))
        return output_array

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCDirichlet(self.N, quad=self.quad, domain=self.domain,
                                     coordinates=self.coors.coordinates)
        return self._bc_basis

    def get_refined(self, N):
        return ShenDirichlet(N,
                             quad=self.quad,
                             domain=self.domain,
                             dtype=self.dtype,
                             padding_factor=self.padding_factor,
                             dealias_direct=self.dealias_direct,
                             coordinates=self.coors.coordinates,
                             bc=self.bc.bc,
                             scaled=self._scaled)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return ShenDirichlet(self.N,
                             quad=self.quad,
                             domain=self.domain,
                             dtype=self.dtype,
                             padding_factor=padding_factor,
                             dealias_direct=dealias_direct,
                             coordinates=self.coors.coordinates,
                             bc=self.bc.bc,
                             scaled=self._scaled)

    def get_unplanned(self):
        return ShenDirichlet(self.N,
                             quad=self.quad,
                             domain=self.domain,
                             dtype=self.dtype,
                             padding_factor=self.padding_factor,
                             dealias_direct=self.dealias_direct,
                             coordinates=self.coors.coordinates,
                             bc=self.bc.bc,
                             scaled=self._scaled)

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
        raise NotImplementedError
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i < self.N-2:
            w = np.arccos(x)
            output_array[:] = np.cos(i*w) - np.cos((i+2)*w)
        elif i == self.N-2:
            output_array[:] = 0.5*(1-x)
        elif i == self.N-1:
            output_array[:] = 0.5*(1+x)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        raise NotImplementedError
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        if i < self.N-2:
            basis = np.zeros(self.shape(True))
            basis[np.array([i, i+2])] = (1, -1)
            basis = n_cheb.Chebyshev(basis)
            if k > 0:
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
            k = np.arange(self.N).astype(float)
            #self._factor = (k+1)**2/(k+3)
            self._factor = (k+1)/(k+3)

    def is_scaled(self):
        """Return True if scaled basis is used, otherwise False"""
        return False

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
        w0 = output.copy()
        self.set_factor_array(w0)
        k = np.arange(self.N).astype(float)
        output[s0] -= self._factor[s0]*w0[s1]
        output[self.si[-2]] = 0
        output[self.si[-1]] = 0

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)
        s0 = self.sl[slice(0, self.N-2)]
        s1 = self.sl[slice(2, self.N)]
        k = np.arange(self.N)
        self.set_factor_array(output_array)
        output_array[s0] = input_array[s0]
        output_array[s1] -= input_array[s0]*self._factor[s0]
        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def eval(self, x, u, output_array=None):
        raise NotImplementedError
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

    def get_refined(self, N):
        return DirichletU(N,
                          quad=self.quad,
                          domain=self.domain,
                          dtype=self.dtype,
                          padding_factor=self.padding_factor,
                          dealias_direct=self.dealias_direct,
                          coordinates=self.coors.coordinates,
                          bc=self.bc.bc,
                          scaled=self._scaled)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return DirichletU(self.N,
                          quad=self.quad,
                          domain=self.domain,
                          dtype=self.dtype,
                          padding_factor=padding_factor,
                          dealias_direct=dealias_direct,
                          coordinates=self.coors.coordinates,
                          bc=self.bc.bc,
                          scaled=self._scaled)

    def get_unplanned(self):
        return DirichletU(self.N,
                          quad=self.quad,
                          domain=self.domain,
                          dtype=self.dtype,
                          padding_factor=self.padding_factor,
                          dealias_direct=self.dealias_direct,
                          coordinates=self.coors.coordinates,
                          bc=self.bc.bc,
                          scaled=self._scaled)


class Heinrichs(CompositeSpace):
    r"""Function space for Dirichlet boundary conditions

    The basis function is

    .. math::

        \phi_k = (1-x^2)T_k

    If scaled is True it is

    .. math::

        \phi_k = (1-x^2)T_k/(k+1)

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
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self._scaled = scaled
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'HH'

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def sympy_basis(self, i=0, x=xp):
        assert i < self.N
        if i < self.N-2:
            f = (1-x**2)*sp.chebyshevt(i, x)
            if self.is_scaled():
                return f/(i+1)
            else:
                return f
        if i == self.N-2:
            return 0.5*(1-x)
        return 0.5*(1+x)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i < self.N-2:
            w = np.arccos(x)
            if self.is_scaled():
                output_array[:] = (1-x**2)*np.cos(i*w)/(i+1)
            else:
                output_array[:] = (1-x**2)*np.cos(i*w)
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
        if self.is_scaled():
            k = np.arange(V.shape[1]).astype(float)
            P[:, 2:-2] = -V[:, :-4]/(k[3:-1]*4) + V[:, 2:-2]/(k[3:-1]*2) - V[:, 4:]/(k[3:-1]*4)
            P[:, 1] = (V[:, 1] - V[:, 3])/8
        else:
            P[:, 2:-2] = -0.25*V[:, :-4] + 0.5*V[:, 2:-2] - 0.25*V[:, 4:]
            P[:, 1] = (V[:, 1] - V[:, 3])/4
        P[:, 0] = (V[:, 0] - V[:, 2])/2
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
        output[self.si[0]] = 0.5*(wk[self.si[0]]-wk[self.si[2]])
        if self.is_scaled():
            k = self.broadcast_to_ndims(np.arange(self.N).astype(float))
            output[self.si[1]] = (wk[self.si[1]]-wk[self.si[3]])/8
            output[self.sl[slice(2, self.N-2)]] *= 1/(2*k[self.sl[slice(3, self.N-1)]])
            output[self.sl[slice(2, self.N-2)]] -= wk[self.sl[slice(0, self.N-4)]]/(4*k[self.sl[slice(3, self.N-1)]])
            output[self.sl[slice(2, self.N-2)]] -= wk[self.sl[slice(4, self.N)]]/(4*k[self.sl[slice(3, self.N-1)]])
        else:
            output[self.si[1]] = 0.25*(wk[self.si[1]]-wk[self.si[3]])
            output[self.sl[slice(2, self.N-2)]] *= 0.5
            output[self.sl[slice(2, self.N-2)]] -= 0.25*wk[self.sl[slice(0, self.N-4)]]
            output[self.sl[slice(2, self.N-2)]] -= 0.25*wk[self.sl[slice(4, self.N)]]
        output[self.si[-2]] = 0
        output[self.si[-1]] = 0

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)
        output_array[self.si[0]] = 0.5*input_array[self.si[0]]
        output_array[self.si[2]] = -0.5*input_array[self.si[0]]
        if self.is_scaled():
            output_array[self.si[1]] = input_array[self.si[1]]/8
            output_array[self.si[3]] = -input_array[self.si[1]]/8
        else:
            output_array[self.si[1]] = input_array[self.si[1]]/4
            output_array[self.si[3]] = -input_array[self.si[1]]/4

        s0 = self.sl[slice(0, self.N-4)]
        s1 = self.sl[slice(2, self.N-2)]
        s2 = self.sl[slice(4, self.N)]
        if self.is_scaled():
            s3 = self.sl[slice(3, self.N-1)]
            k = self.broadcast_to_ndims(np.arange(self.N))
            w0 = input_array[s1]/k[s3]
        else:
            w0 = input_array[s1]
        output_array[s0] -= 0.25*w0
        output_array[s1] += 0.5*w0
        output_array[s2] -= 0.25*w0
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

    def get_refined(self, N):
        return Heinrichs(N,
                         quad=self.quad,
                         domain=self.domain,
                         dtype=self.dtype,
                         padding_factor=self.padding_factor,
                         dealias_direct=self.dealias_direct,
                         coordinates=self.coors.coordinates,
                         bc=self.bc.bc,
                         scaled=self._scaled)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return Heinrichs(self.N,
                         quad=self.quad,
                         domain=self.domain,
                         dtype=self.dtype,
                         padding_factor=padding_factor,
                         dealias_direct=dealias_direct,
                         coordinates=self.coors.coordinates,
                         bc=self.bc.bc,
                         scaled=self._scaled)

    def get_unplanned(self):
        return Heinrichs(self.N,
                         quad=self.quad,
                         domain=self.domain,
                         dtype=self.dtype,
                         padding_factor=self.padding_factor,
                         dealias_direct=self.dealias_direct,
                         coordinates=self.coors.coordinates,
                         bc=self.bc.bc,
                         scaled=self._scaled)


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
    def __init__(self, N, quad="GC", mean=0, bc=(0, 0), domain=(-1., 1.), dtype=float, padding_factor=1,
                 dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self.mean = mean
        self._factor = np.zeros(0)
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

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

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def _composite(self, V, argument=0):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(float)
        P[:, :-2] = V[:, :-2] - (k[:-2]/(k[:-2]+2))**2*V[:, 2:]
        if argument == 1: # if trial function
            P[:, -1] = 0.5*V[:, 1] + 1/8*V[:, 2]    # x = +1
            P[:, -2] = 0.5*V[:, 1] - 1/8*V[:, 2]    # x = -1
        return P

    def sympy_basis(self, i=0, x=xp):
        if 0 < i < self.N-2:
            return sp.chebyshevt(i, x) - (i/(i+2))**2*sp.chebyshevt(i+2, x)
        return 0

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        w = np.arccos(x)
        output_array[:] = np.cos(i*w) - (i*1./(i+2))**2*np.cos((i+2)*w)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[np.array([i, i+2])] = (1, -(i*1./(i+2))**2)
        basis = n_cheb.Chebyshev(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def set_factor_array(self, v):
        """Set intermediate factor arrays"""
        if not self._factor.shape == v.shape:
            k = np.arange(self.N-2).astype(float)
            self._factor = self.broadcast_to_ndims((k/(k+2))**2)

    def _evaluate_scalar_product(self, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            if self.use_fixed_gauge:
                self.scalar_product.output_array[self.si[0]] = self.mean*np.pi
            self.scalar_product.output_array[self.sl[slice(-2, None)]] = 0
            return
        #output = self.CT.scalar_product(fast_transform=True)
        Orthogonal._evaluate_scalar_product(self, True)
        output = self.scalar_product.output_array
        self.set_factor_array(output)
        sm2 = self.sl[slice(0, self.N-2)]
        s2p = self.sl[slice(2, self.N)]
        output[sm2] -= self._factor * output[s2p]
        if self.use_fixed_gauge:
            output[self.si[0]] = self.mean*np.pi
        output[self.sl[slice(-2, None)]] = 0

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)
        s0 = self.sl[slice(0, self.N-2)]
        s1 = self.sl[slice(2, self.N)]
        self.set_factor_array(input_array)
        output_array[s0] = input_array[s0]
        output_array[s1] -= self._factor*input_array[s0]
        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        #if self.use_fixed_gauge:
        #    return slice(1, self.N-2)
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
        output_array += u[-2]*(0.5*x-1/8*(2*x**2-1)) + u[-2]*(0.5*x+1/8*(2*x**2-1))
        return output_array

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
    def __init__(self, N, quad="GC", mean=0, bc=(0, 0), domain=(-1., 1.), dtype=float, padding_factor=1,
                 dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self.mean = mean
        self._factor = np.zeros(0)
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

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

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def _composite(self, V, argument=0):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(float)
        P[:, 3:-2] = -V[:, 1:-4]/k[1:-4]**2 + 2*V[:, 3:-2]/k[3:-2]**2 - V[:, 5:]/k[5:]**2
        P[:, 2] = V[:, 2]/4 - V[:, 4]/16
        P[:, 1] = V[:, 1] - V[:, 3]/9
        P[:, 0] = V[:, 0]
        if argument == 1: # if trial function
            P[:, -1] = 0.5*V[:, 1] + 1/8*V[:, 2]    # x = +1
            P[:, -2] = 0.5*V[:, 1] - 1/8*V[:, 2]    # x = -1
        return P

    def sympy_basis(self, i=0, x=xp):
        raise NotImplementedError
        if 0 < i < self.N-2:
            return sp.chebyshevt(i, x) - (i/(i+2))**2*sp.chebyshevt(i+2, x)
        return 0

    def evaluate_basis(self, x, i=0, output_array=None):
        raise NotImplementedError
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        w = np.arccos(x)
        output_array[:] = np.cos(i*w) - (i*1./(i+2))**2*np.cos((i+2)*w)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        raise NotImplementedError
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[np.array([i, i+2])] = (1, -(i*1./(i+2))**2)
        basis = n_cheb.Chebyshev(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def set_factor_array(self, v):
        """Set intermediate factor arrays"""
        if not self._factor.shape == v.shape:
            k = np.arange(self.N).astype(float)
            self._factor = self.broadcast_to_ndims(k**2)

    def _evaluate_scalar_product(self, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            self.scalar_product.output_array[self.si[0]] = self.mean*np.pi
            self.scalar_product.output_array[self.sl[slice(-2, None)]] = 0
            return
        Orthogonal._evaluate_scalar_product(self, True)
        output = self.scalar_product.output_array
        wk = output.copy()
        self.set_factor_array(output)
        k2 = self._factor
        if self.use_fixed_gauge:
            output[self.si[0]] = self.mean*np.pi
        output[self.si[1]] = wk[self.si[1]] - wk[self.si[3]]/9
        output[self.si[2]] = wk[self.si[2]]/4 - wk[self.si[4]]/16
        s1 = self.sl[slice(1, self.N-4)]
        s2 = self.sl[slice(3, self.N-2)]
        s3 = self.sl[slice(5, self.N)]
        output[s2] = -wk[s1]/k2[s1]
        output[s2] += 2*wk[s2]/k2[s2]
        output[s2] -= wk[s3]/k2[s3]
        output[self.sl[slice(-2, None)]] = 0

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)
        s0 = self.sl[slice(1, self.N-4)]
        s1 = self.sl[slice(3, self.N-2)]
        s2 = self.sl[slice(5, self.N)]
        self.set_factor_array(input_array)
        k2 = self._factor
        output_array[self.si[0]] = input_array[self.si[0]]
        output_array[self.si[1]] = input_array[self.si[1]]
        output_array[self.si[2]] = input_array[self.si[2]]/4
        output_array[self.si[3]] = -input_array[self.si[1]]/9
        output_array[self.si[4]] = -input_array[self.si[2]]/16
        output_array[s0] -= input_array[s1]/k2[s0]
        output_array[s1] += input_array[s1]*2/k2[s1]
        output_array[s2] -= input_array[s1]/k2[s2]
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
        self._bc_basis = BCNeumann(self.N, quad=self.quad, domain=self.domain,
                                   coordinates=self.coors.coordinates)
        return self._bc_basis

    def get_refined(self, N):
        return CombinedShenNeumann(N,
                           quad=self.quad,
                           bc=self.bc.bc,
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
                           domain=self.domain,
                           dtype=self.dtype,
                           padding_factor=self.padding_factor,
                           dealias_direct=self.dealias_direct,
                           coordinates=self.coors.coordinates,
                           mean=self.mean)


class MikNeumann(CompositeSpace):
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
    def __init__(self, N, quad="GC", mean=0, bc=(0, 0), domain=(-1., 1.), dtype=float, padding_factor=1,
                 dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self.mean = mean
        self._factor = np.zeros(0)
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

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

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def _composite(self, V, argument=0):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(float)
        P[:, 3:-2] = -V[:, 1:-4]/k[1:-4] + 2*V[:, 3:-2]/k[3:-2] - V[:, 5:]/k[5:]
        P[:, 2] = V[:, 2] - V[:, 4]/4
        P[:, 1] = 3*V[:, 1] - V[:, 3]/3
        P[:, 0] = V[:, 0]
        if argument == 1: # if trial function
            P[:, -1] = 0.5*V[:, 1] + 1/8*V[:, 2]    # x = +1
            P[:, -2] = 0.5*V[:, 1] - 1/8*V[:, 2]    # x = -1
        return P

    def sympy_basis(self, i=0, x=xp):
        if i == 0:
            return sp.chebyshevt(i, x)
        elif i == 1:
            return 3*sp.chebyshevt(i, x) - sp.chebyshevt(i+2)/3
        elif i == 2:
            return sp.chebyshevt(i, x) - sp.chebyshevt(i+2)/4
        else:
            return -sp.chebyshevt(i-2, x)/(i-2) + 2*sp.chebyshevt(i, x)/i - sp.chebyshevt(i+2, x)/(i+2)
        return 0

    def evaluate_basis(self, x, i=0, output_array=None):
        raise NotImplementedError
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        w = np.arccos(x)
        if i == 0:
            output_array[:] = 1
        elif i == 1:
            output_array[:] = 3*np.cos(i*w) - np.cos((i+2)*w)/3
        elif i == 2:
            output_array[:] = np.cos(i*w) - np.cos((i+2)*w)/4
        else:
            output_array[:] = -np.cos((i-2)*w)/(i-2) + 2*np.cos(i*w)/i - np.cos((i+2)*w)/(i+2)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        raise NotImplementedError
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        if i == 0:
            basis[np.array([i])] = 1
        elif i == 1:
            basis[np.array([i, i+2])] = (3, -1/3)
        elif i == 2:
            basis[np.array([i, i+2])] = (1, -1/4)
        else:
            basis[np.array([i-2, i, i+2])] = (1/(i-2), 2/i, -1/(i+2))

        basis = n_cheb.Chebyshev(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def set_factor_array(self, v):
        """Set intermediate factor arrays"""
        if not self._factor.shape == v.shape:
            k = np.arange(self.N).astype(float)
            self._factor = self.broadcast_to_ndims(k)

    def _evaluate_scalar_product(self, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            self.scalar_product.output_array[self.si[0]] = self.mean*np.pi
            self.scalar_product.output_array[self.sl[slice(-2, None)]] = 0
            return
        Orthogonal._evaluate_scalar_product(self, True)
        output = self.scalar_product.output_array
        wk = output.copy()
        self.set_factor_array(output)
        k2 = self._factor
        if self.use_fixed_gauge:
            output[self.si[0]] = self.mean*np.pi
        output[self.si[1]] = 3*wk[self.si[1]] - wk[self.si[3]]/3
        output[self.si[2]] = wk[self.si[2]] - wk[self.si[4]]/4
        s1 = self.sl[slice(1, self.N-4)]
        s2 = self.sl[slice(3, self.N-2)]
        s3 = self.sl[slice(5, self.N)]
        output[s2] = -wk[s1]/k2[s1]
        output[s2] += 2*wk[s2]/k2[s2]
        output[s2] -= wk[s3]/k2[s3]
        output[self.sl[slice(-2, None)]] = 0

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)
        s0 = self.sl[slice(1, self.N-4)]
        s1 = self.sl[slice(3, self.N-2)]
        s2 = self.sl[slice(5, self.N)]
        self.set_factor_array(input_array)
        k2 = self._factor
        output_array[self.si[0]] = input_array[self.si[0]]
        output_array[self.si[1]] = 3*input_array[self.si[1]]
        output_array[self.si[2]] = input_array[self.si[2]]
        output_array[self.si[3]] = -input_array[self.si[1]]/3
        output_array[self.si[4]] = -input_array[self.si[2]]/4

        output_array[s0] -= input_array[s1]/k2[s0]
        output_array[s1] += input_array[s1]*2/k2[s1]
        output_array[s2] -= input_array[s1]/k2[s2]

        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        #if self.use_fixed_gauge:
        #    return slice(1, self.N-2)
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
                 padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self._factor1 = np.zeros(0)
        self._factor2 = np.zeros(0)
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    @staticmethod
    def short_name():
        return 'SB'

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def _composite(self, V, argument=0):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(float)[:-4]
        P[:, :-4] = V[:, :-4] - (2*(k+2)/(k+3))*V[:, 2:-2] + ((k+1)/(k+3))*V[:, 4:]
        if argument == 1: # if trial function
            P[:, -4:] = np.tensordot(V[:, :4], BCBiharmonic.coefficient_matrix(), (1, 1))
        return P

    def sympy_basis(self, i=0, x=xp):
        if i < self.N-4:
            f = sp.chebyshevt(i, x) - (2*(i+2)/(i+3))*sp.chebyshevt(i+2, x) + (i+1)/(i+3)*sp.chebyshevt(i+4, x)
        else:
            f = BCBiharmonic.coefficient_matrix()[i]*np.array([sp.chebyshevt(j, x) for j in range(4)])
        return f

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        w = np.arccos(x)
        output_array[:] = np.cos(i*w) - (2*(i+2.)/(i+3.))*np.cos((i+2)*w) + ((i+1.)/(i+3.))*np.cos((i+4)*w)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[np.array([i, i+2, i+4])] = (1, -(2*(i+2.)/(i+3.)), ((i+1.)/(i+3.)))
        basis = n_cheb.Chebyshev(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def set_factor_arrays(self, v):
        """Set intermediate factor arrays"""
        s = self.sl[self.slice()]
        if not self._factor1.shape == v[s].shape:
            k = self.wavenumbers().astype(float)
            self._factor1 = (-2*(k+2)/(k+3)).astype(float)
            self._factor2 = ((k+1)/(k+3)).astype(float)

    def _evaluate_scalar_product(self, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            self.scalar_product.output_array[self.sl[slice(-4, None)]] = 0
            return
        Orthogonal._evaluate_scalar_product(self, True)
        output = self.scalar_product.output_array
        Tk = work[(output, 0, True)]
        Tk[...] = output
        self.set_factor_arrays(Tk)
        s = self.sl[self.slice()]
        s2 = self.sl[slice(2, self.N-2)]
        s4 = self.sl[slice(4, self.N)]
        output[s] += self._factor1*Tk[s2]
        output[s] += self._factor2*Tk[s4]
        self.bc.set_boundary_dofs(output)
        output[self.sl[slice(-4, None)]] = 0

    #@optimizer
    def set_w_hat(self, w_hat, fk, f1, f2):
        """Return intermediate w_hat array"""
        s = self.sl[self.slice()]
        s2 = self.sl[slice(2, self.N-2)]
        s4 = self.sl[slice(4, self.N)]
        w_hat[s] = fk[s]
        w_hat[s2] += f1*fk[s]
        w_hat[s4] += f2*fk[s]
        return w_hat

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)
        self.set_factor_arrays(input_array)
        output_array = self.set_w_hat(output_array, input_array, self._factor1, self._factor2)
        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        return slice(0, self.N-4)

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_arrays(w_hat)
        output_array[:] = n_cheb.chebval(x, u[:-4])
        w_hat[2:-2] = self._factor1*u[:-4]
        output_array += n_cheb.chebval(x, w_hat[:-2])
        w_hat[4:] = self._factor2*u[:-4]
        w_hat[:4] = 0
        output_array += n_cheb.chebval(x, w_hat)
        return output_array

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCBiharmonic(self.N, quad=self.quad, domain=self.domain,
                                      coordinates=self.coors.coordinates)
        return self._bc_basis

    def get_refined(self, N):
        return ShenBiharmonic(N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return ShenBiharmonic(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=padding_factor,
                              dealias_direct=dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

    def get_unplanned(self):
        return ShenBiharmonic(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)


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
        return sp.chebyshevt(i, x) - (i/(i+2))**2*(i**2-1)/((i+2)**2-1)*sp.chebyshevt(i+2, x)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        w = np.arccos(x)
        output_array[:] = np.cos(i*w) - (i*1./(i+2))**2*(i**2-1.)/((i+2)**2-1.)*np.cos((i+2)*w)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
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
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self._scaled = scaled
        self._factor = np.ones(1)
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'UpperDirichlet'

    @staticmethod
    def short_name():
        return 'UD'

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def _composite(self, V, argument=0):
        P = np.zeros_like(V)
        P[:, :-1] = V[:, :-1] - V[:, 1:]
        if argument == 1: # if trial function
            P[:, -1] = (V[:, 0] + V[:, 1])/2    # x = +1
        return P

    def sympy_basis(self, i=0, x=xp):
        if i < self.N-1:
            return sp.chebyshevt(i, x) - sp.chebyshevt(i+1, x)
        assert i == self.N-1
        return 0.5*(1+x)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i < self.N-1:
            w = np.arccos(x)
            output_array[:] = np.cos(i*w) - np.cos((i+1)*w)
        elif i == self.N-1:
            output_array[:] = 0.5*(1+x)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        if i < self.N-1:
            basis = np.zeros(self.shape(True))
            basis[np.array([i, i+1])] = (1, -1)
            basis = n_cheb.Chebyshev(basis)
            if k > 0:
                basis = basis.deriv(k)
            output_array[:] = basis(x)
        else:
            if k == 1:
                output_array[:] = 0.5
            else:
                output_array[:] = 0
        return output_array

    def is_scaled(self):
        """Return True if scaled basis is used, otherwise False"""
        return False

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCUpperDirichlet(self.N, quad=self.quad, domain=self.domain,
                                          coordinates=self.coors.coordinates)
        return self._bc_basis

    def _evaluate_scalar_product(self, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            self.scalar_product.output_array[self.si[-1]] = 0
            return
        Orthogonal._evaluate_scalar_product(self, True)
        output = self.scalar_product.output_array
        s0 = self.sl[slice(0, -1)]
        s1 = self.sl[slice(1, None)]
        output[s0] -= output[s1]
        output[self.si[-1]] = 0

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)
        s0 = self.sl[slice(0, -1)]
        s1 = self.sl[slice(1, None)]
        output_array[s0] = input_array[s0]
        output_array[s1] -= input_array[s0]
        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        return slice(0, self.N-1)

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        output_array[:] = chebval(x, u[:-1])
        w_hat[1:] = u[:-1]
        output_array -= chebval(x, w_hat)
        output_array += 0.5*u[-1]*(1+x)
        return output_array

    def get_refined(self, N):
        return UpperDirichlet(N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc,
                              scaled=self._scaled)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return UpperDirichlet(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=padding_factor,
                              dealias_direct=dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc,
                              scaled=self._scaled)

    def get_unplanned(self):
        return UpperDirichlet(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc,
                              scaled=self._scaled)


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
    def __init__(self, N, quad="GC", domain=(-1., 1.), dtype=float,
                 padding_factor=1, dealias_direct=False, coordinates=None):
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


class HeinrichsBiharmonic(CompositeSpace):
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
                 padding_factor=1, dealias_direct=False, coordinates=None):
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    @staticmethod
    def short_name():
        return 'HB'

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def _composite(self, V, argument=0):
        P = np.zeros_like(V)
        P[:, 4:-4] = (V[:, :-8]-2*V[:, 2:-6]+6*V[:, 4:-4]-4*V[:, 6:-2]+V[:, 8:])/16
        P[:, 3] = (-7*V[:, 1]+6*V[:, 3]-4*V[:, 5]+V[:, 7])/16
        P[:, 2] = (-4*V[:, 0]+7*V[:, 2]-4*V[:, 4]+V[:, 6])/16
        P[:, 1] = (2*V[:, 1]-3*V[:, 3]+V[:, 5])/16
        P[:, 0] = (6*V[:, 0]-8*V[:, 2]+2*V[:, 4])/16
        if argument == 1: # if trial function
            P[:, -4:] = np.tensordot(V[:, :4], BCBiharmonic.coefficient_matrix(), (1, 1))
        return P

    def sympy_basis(self, i=0, x=xp):
        if i < self.N-4:
            f = (1-x**2)**2*sp.chebyshevt(i, x)
        else:
            f = BCBiharmonic.coefficient_matrix()[i]*np.array([sp.chebyshevt(j, x) for j in range(4)])
        return f

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        w = np.arccos(x)
        output_array[:] = (1-x**2)**2*np.cos(i*w)
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

    def _evaluate_scalar_product(self, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            self.scalar_product.output_array[self.sl[slice(-4, None)]] = 0
            return
        Orthogonal._evaluate_scalar_product(self, True)
        output = self.scalar_product.output_array
        wk = output.copy()

        output[self.si[0]] = (6*wk[self.si[0]]-8*wk[self.si[2]]+2*wk[self.si[4]])/16
        output[self.si[1]] = (2*wk[self.si[1]]-3*wk[self.si[3]]+wk[self.si[5]])/16
        output[self.si[2]] = (-4*wk[self.si[0]]+7*wk[self.si[2]]-4*wk[self.si[4]]+wk[self.si[6]])/16
        output[self.si[3]] = (-7*wk[self.si[1]]+6*wk[self.si[3]]-4*wk[self.si[5]]+wk[self.si[7]])/16

        output[self.sl[slice(4, self.N-4)]] *= 3/8
        output[self.sl[slice(4, self.N-4)]] -= wk[self.sl[slice(2, self.N-6)]]/8
        output[self.sl[slice(4, self.N-4)]] += wk[self.sl[slice(0, self.N-8)]]/16
        output[self.sl[slice(4, self.N-4)]] -= wk[self.sl[slice(6, self.N-2)]]/4
        output[self.sl[slice(4, self.N-4)]] += wk[self.sl[slice(8, self.N)]]/16
        output[self.sl[slice(-4, None)]] = 0

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)
        output_array[self.si[0]] = 3/8*input_array[self.si[0]]-1/4*input_array[self.si[2]]
        output_array[self.si[1]] = 1/8*input_array[self.si[1]]-7/16*input_array[self.si[3]]
        output_array[self.si[2]] = -1/2*input_array[self.si[0]]+7/16*input_array[self.si[2]]
        output_array[self.si[3]] = -3/16*input_array[self.si[1]]-7/16*input_array[self.si[3]]
        output_array[self.si[4]] = 1/8*input_array[self.si[0]]-1/4*input_array[self.si[2]]
        output_array[self.si[5]] = 1/16*input_array[self.si[1]]-1/4*input_array[self.si[3]]
        output_array[self.si[6]] = 1/16*input_array[self.si[2]]
        output_array[self.si[7]] = 1/16*input_array[self.si[3]]
        s0 = self.sl[slice(0, self.N-8)]
        s1 = self.sl[slice(2, self.N-6)]
        s2 = self.sl[slice(4, self.N-4)]
        s3 = self.sl[slice(6, self.N-2)]
        s4 = self.sl[slice(8, self.N)]
        output_array[s0] += input_array[s2]/16
        output_array[s1] -= input_array[s2]/8
        output_array[s2] += input_array[s2]*(3/8)
        output_array[s3] -= input_array[s2]/4
        output_array[s4] += input_array[s2]/16
        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        return slice(0, self.N-4)

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
        self._bc_basis = BCBiharmonic(self.N, quad=self.quad, domain=self.domain,
                                      coordinates=self.coors.coordinates)
        return self._bc_basis

    def get_refined(self, N):
        return HeinrichsBiharmonic(N,
                                   quad=self.quad,
                                   domain=self.domain,
                                   dtype=self.dtype,
                                   padding_factor=self.padding_factor,
                                   dealias_direct=self.dealias_direct,
                                   coordinates=self.coors.coordinates,
                                   bc=self.bc.bc)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return HeinrichsBiharmonic(self.N,
                                   quad=self.quad,
                                   domain=self.domain,
                                   dtype=self.dtype,
                                   padding_factor=padding_factor,
                                   dealias_direct=dealias_direct,
                                   coordinates=self.coors.coordinates,
                                   bc=self.bc.bc)

    def get_unplanned(self):
        return HeinrichsBiharmonic(self.N,
                                   quad=self.quad,
                                   domain=self.domain,
                                   dtype=self.dtype,
                                   padding_factor=self.padding_factor,
                                   dealias_direct=self.dealias_direct,
                                   coordinates=self.coors.coordinates,
                                   bc=self.bc.bc)


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
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                                padding_factor=padding_factor, dealias_direct=dealias_direct, coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self._scaled = scaled
        self._factor1 = np.zeros(0)
        self._factor2 = np.zeros(0)
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'DirichletNeumann'

    @staticmethod
    def short_name():
        return 'DN'

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def set_factor_arrays(self, v):
        """Set intermediate factor arrays"""
        s = self.sl[self.slice()]
        if not self._factor1.shape == v[s].shape:
            k = self.wavenumbers().astype(float)
            self._factor1 = ((-k**2 + (k+2)**2)/((k+1)**2 + (k+2)**2)).astype(float)
            self._factor2 = ((-k**2 - (k+1)**2)/((k+1)**2 + (k+2)**2)).astype(float)

    def _composite(self, V, argument=0):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(float)[:-2]
        P[:, :-2] = (V[:, :-2]
                     + ((-k**2 + (k+2)**2)/((k+1)**2 + (k+2)**2))*V[:, 1:-1]
                     + ((-k**2 - (k+1)**2)/((k+1)**2 + (k+2)**2))*V[:, 2:])
        if argument == 1:
            P[:, -2] = V[:, 0]
            P[:, -1] = V[:, 0]+V[:, 1]
        return P

    def sympy_basis(self, i=0, x=xp):
        assert i < self.N-2
        return (sp.chebyshevt(i, x)
                + ((-i**2 + (i+2)**2) / ((i+1)**2 + (i+2)**2))*sp.chebyshevt(i+1, x)
                + ((-i**2 - (i+1)**2) / ((i+1)**2 + (i+2)**2))*sp.chebyshevt(i+2, x))

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        w = np.arccos(x)
        output_array[:] = (np.cos(i*w)
                           + ((-i**2 + (i+2)**2)/((i+1)**2 + (i+2)**2))*np.cos((i+1)*w)
                           + ((-i**2 - (i+1)**2)/((i+1)**2 + (i+2)**2))*np.cos((i+2)*w))
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[np.array([i, i+1, i+2])] = (1,((-i**2 + (i+2)**2) / ((i+1)**2 + (i+2)**2)), ((-i**2 - (i+1)**2) / ((i+1)**2 + (i+2)**2)))
        basis = n_cheb.Chebyshev(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def is_scaled(self):
        """Return True if scaled basis is used, otherwise False"""
        return False

    def set_w_hat(self, w_hat, fk, f1, f2):
        """Return intermediate w_hat array"""
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(1, -1)]
        s2 = self.sl[slice(2, None)]
        w_hat[s0] = fk[s0]
        w_hat[s1] += f1*fk[s0]
        w_hat[s2] += f2*fk[s0]
        return w_hat

    def _evaluate_scalar_product(self, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            self.scalar_product.output_array[self.sl[slice(-2, None)]] = 0
            return
        Orthogonal._evaluate_scalar_product(self, True)
        output = self.scalar_product.output_array
        Tk = work[(output, 0, True)]
        Tk[...] = output
        self.set_factor_arrays(Tk)
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(1, -1)]
        s2 = self.sl[slice(2, None)]
        output[s0] += self._factor1*Tk[s1]
        output[s0] += self._factor2*Tk[s2]
        output[self.sl[slice(-2, None)]] = 0

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)
        self.set_factor_arrays(input_array)
        output_array = self.set_w_hat(output_array, input_array, self._factor1, self._factor2)
        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_arrays(w_hat)
        output_array[:] = chebval(x, u[:-2])
        w_hat[1:-1] = self._factor1*u[:-2]
        output_array += chebval(x, w_hat)
        w_hat[2:] = self._factor2*u[:-2]
        w_hat[:2] = 0
        output_array += chebval(x, w_hat)
        output_array += u[-2] + u[-1]*(1+x)
        return output_array

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCDirichletNeumann(self.N, quad=self.quad, domain=self.domain,
                                            coordinates=self.coors.coordinates)
        return self._bc_basis

    def get_refined(self, N):
        return self.__class__(N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return self.__class__(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=padding_factor,
                              dealias_direct=dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

    def get_unplanned(self):
        return self.__class__(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)


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
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                                padding_factor=padding_factor, dealias_direct=dealias_direct, coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self._scaled = scaled
        self._factor1 = np.zeros(0)
        self._factor2 = np.zeros(0)
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'NeumannDirichlet'

    @staticmethod
    def short_name():
        return 'ND'

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def set_factor_arrays(self, v):
        """Set intermediate factor arrays"""
        s = self.sl[self.slice()]
        if not self._factor1.shape == v[s].shape:
            k = self.wavenumbers().astype(float)
            self._factor1 = (-(-k**2 + (k+2)**2)/((k+1)**2 + (k+2)**2)).astype(float)
            self._factor2 = ((-k**2 - (k+1)**2)/((k+1)**2 + (k+2)**2)).astype(float)

    def _composite(self, V, argument=0):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(float)[:-2]
        P[:, :-2] = (V[:, :-2]
                     - ((-k**2 + (k+2)**2)/((k+1)**2 + (k+2)**2))*V[:, 1:-1]
                     + ((-k**2 - (k+1)**2)/((k+1)**2 + (k+2)**2))*V[:, 2:])
        if argument == 1:
            P[:, -2] = V[:, 0]-0.6*V[:, 1]-0.4*V[:, 2]
            P[:, -1] = V[:, 0]
        return P

    def sympy_basis(self, i=0, x=xp):
        assert i < self.N-2
        return (sp.chebyshevt(i, x)
                - ((-i**2 + (i+2)**2) / ((i+1)**2 + (i+2)**2))*sp.chebyshevt(i+1, x)
                + ((-i**2 - (i+1)**2) / ((i+1)**2 + (i+2)**2))*sp.chebyshevt(i+2, x))

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        w = np.arccos(x)
        output_array[:] = (np.cos(i*w)
                           - ((-i**2 + (i+2)**2)/((i+1)**2 + (i+2)**2))*np.cos((i+1)*w)
                           + ((-i**2 - (i+1)**2)/((i+1)**2 + (i+2)**2))*np.cos((i+2)*w))
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[np.array([i, i+1, i+2])] = (1,-((-i**2 + (i+2)**2) / ((i+1)**2 + (i+2)**2)), ((-i**2 - (i+1)**2) / ((i+1)**2 + (i+2)**2)))
        basis = n_cheb.Chebyshev(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def is_scaled(self):
        """Return True if scaled basis is used, otherwise False"""
        return False

    def set_w_hat(self, w_hat, fk, f1, f2):
        """Return intermediate w_hat array"""
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(1, -1)]
        s2 = self.sl[slice(2, None)]
        w_hat[s0] = fk[s0]
        w_hat[s1] += f1*fk[s0]
        w_hat[s2] += f2*fk[s0]
        return w_hat

    def _evaluate_scalar_product(self, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            self.scalar_product.output_array[self.sl[slice(-2, None)]] = 0
            return
        Orthogonal._evaluate_scalar_product(self, True)
        output = self.scalar_product.output_array
        Tk = work[(output, 0, True)]
        Tk[...] = output
        self.set_factor_arrays(Tk)
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(1, -1)]
        s2 = self.sl[slice(2, None)]
        output[s0] += self._factor1*Tk[s1]
        output[s0] += self._factor2*Tk[s2]
        output[self.sl[slice(-2, None)]] = 0

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)
        self.set_factor_arrays(input_array)
        output_array = self.set_w_hat(output_array, input_array, self._factor1, self._factor2)
        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_arrays(w_hat)
        output_array[:] = chebval(x, u[:-2])
        w_hat[1:-1] = self._factor1*u[:-2]
        output_array += chebval(x, w_hat)
        w_hat[2:] = self._factor2*u[:-2]
        w_hat[:2] = 0
        output_array += chebval(x, w_hat)
        output_array += u[-1] + u[-2]*(1-0.6*x-0.4*(2*x**2-1))
        return output_array

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCNeumannDirichlet(self.N, quad=self.quad, domain=self.domain,
                                            coordinates=self.coors.coordinates)
        return self._bc_basis

    def get_refined(self, N):
        return self.__class__(N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return self.__class__(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=padding_factor,
                              dealias_direct=dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

    def get_unplanned(self):
        return self.__class__(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

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
        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                                padding_factor=padding_factor, dealias_direct=dealias_direct, coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self._scaled = scaled
        self._factor1 = np.zeros(0)
        self._factor2 = np.zeros(0)
        self._bc_basis = None
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'UpperDirichletNeumann'

    @staticmethod
    def short_name():
        return 'US'

    @property
    def has_nonhomogeneous_bcs(self):
        return self.bc.has_nonhomogeneous_bcs()

    def set_factor_arrays(self, v):
        """Set intermediate factor arrays"""
        s = self.sl[self.slice()]
        if not self._factor1.shape == v[s].shape:
            k = self.wavenumbers().astype(float)
            self._factor1 = (-4*(k+1)/(2*k+3)).astype(float)
            self._factor2 = ((2*k+1)/(2*k+3)).astype(float)

    def _composite(self, V, argument=0):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(float)[:-2]
        P[:, :-2] = (V[:, :-2]
                     + (-4*(k+1)/(2*k+3))*V[:, 1:-1]
                     + ((2*k+1)/(2*k+3))*V[:, 2:])
        if argument == 1:
            P[:, -2] = V[:, 0]
            P[:, -1] = V[:, 0]-5/3*V[:, 1]+2/3*V[:, 2]
        return P

    def sympy_basis(self, i=0, x=xp):
        assert i < self.N-2
        return (sp.chebyshevt(i, x)
                + (-4*(i+1)/(2*i+3))*sp.chebyshevt(i+1, x)
                + ((2*k+1)/(2*k+3))*sp.chebyshevt(i+2, x))

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        w = np.arccos(x)
        output_array[:] = (np.cos(i*w)
                           + (-4*(i+1)/(2*i+3))*np.cos((i+1)*w)
                           + ((2*i+1)/(2*i+3))*np.cos((i+2)*w))
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = np.atleast_1d(x)
        basis = np.zeros(self.shape(True))
        basis[np.array([i, i+1, i+2])] = (1, -4*(i+1)/(2*i+3), (2*i+1)/(2*i+3))
        basis = n_cheb.Chebyshev(basis)
        if k > 0:
            basis = basis.deriv(k)
        output_array[:] = basis(x)
        return output_array

    def is_scaled(self):
        """Return True if scaled basis is used, otherwise False"""
        return False

    def set_w_hat(self, w_hat, fk, f1, f2):
        """Return intermediate w_hat array"""
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(1, -1)]
        s2 = self.sl[slice(2, None)]
        w_hat[s0] = fk[s0]
        w_hat[s1] += f1*fk[s0]
        w_hat[s2] += f2*fk[s0]
        return w_hat

    def _evaluate_scalar_product(self, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            self.scalar_product.output_array[self.sl[slice(-2, None)]] = 0
            return
        Orthogonal._evaluate_scalar_product(self, True)
        output = self.scalar_product.output_array
        Tk = work[(output, 0, True)]
        Tk[...] = output
        self.set_factor_arrays(Tk)
        s0 = self.sl[slice(0, -2)]
        s1 = self.sl[slice(1, -1)]
        s2 = self.sl[slice(2, None)]
        output[s0] += self._factor1*Tk[s1]
        output[s0] += self._factor2*Tk[s2]
        output[self.sl[slice(-2, None)]] = 0

    def to_ortho(self, input_array, output_array=None):
        if output_array is None:
            output_array = np.zeros_like(input_array)
        else:
            output_array.fill(0)
        self.set_factor_arrays(input_array)
        output_array = self.set_w_hat(output_array, input_array, self._factor1, self._factor2)
        self.bc.add_to_orthogonal(output_array, input_array)
        return output_array

    def slice(self):
        return slice(0, self.N-2)

    def eval(self, x, u, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=self.dtype)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0, True)]
        self.set_factor_arrays(w_hat)
        output_array[:] = chebval(x, u[:-2])
        w_hat[1:-1] = self._factor1*u[:-2]
        output_array += chebval(x, w_hat)
        w_hat[2:] = self._factor2*u[:-2]
        w_hat[:2] = 0
        output_array += chebval(x, w_hat)
        output_array += u[-2] + u[-1]*(1-5/3*x+2/3*(2*x**2-1))
        return output_array

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCUpperDirichletNeumann(self.N, quad=self.quad, domain=self.domain,
                                                 coordinates=self.coors.coordinates)
        return self._bc_basis

    def get_refined(self, N):
        return self.__class__(N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return self.__class__(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=padding_factor,
                              dealias_direct=dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)

    def get_unplanned(self):
        return self.__class__(self.N,
                              quad=self.quad,
                              domain=self.domain,
                              dtype=self.dtype,
                              padding_factor=self.padding_factor,
                              dealias_direct=self.dealias_direct,
                              coordinates=self.coors.coordinates,
                              bc=self.bc.bc)


class MixedTU(CompositeSpace):
    r"""Dirichlet class

    .. math::

        \phi_k = (1-x^2) T^{''}_{k+2}, \quad k=0, 1, \ldots, N-3

    or equally

    .. math::

        \phi_k = (k+2) U_{k+2} - (k+2)(k+3)T_{k+2} , \quad k=0, 1, \ldots, N-3

    The two formulations given above are identical. Since (k+2) is
    a factor in both terms of the latter, we can simply neglect it.
    Choose to neglect k+2 using scaled=True.

    Note
    ----
    Use DirichletU class instead! The implementation is better and
    this one will be removed.

    """
    def __init__(self, N, quad="GC", bc=(0, 0), domain=(-1., 1.), dtype=float, scaled=False,
                 padding_factor=1, dealias_direct=False, coordinates=None):
        assert quad == "GC"
        self._dst_fwd = functools.partial(fftw.dstn, type=2)
        self._dst_bck = functools.partial(fftw.dstn, type=3)

        CompositeSpace.__init__(self, N, quad=quad, domain=domain, dtype=dtype,
                                padding_factor=padding_factor, dealias_direct=dealias_direct,
                                coordinates=coordinates)
        from shenfun.tensorproductspace import BoundaryValues
        self._factor = np.ones(1)
        self._bc_basis = None
        self._scaled = scaled
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    @staticmethod
    def short_name():
        return 'TU'

    def is_scaled(self):
        """Return True if scaled basis is used, otherwise False"""
        return self._scaled

    def plan(self, shape, axis, dtype, options):
        CompositeSpace.plan(self, shape, axis, dtype, options)

        if shape in (0, (0,)):
            return

        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[-1]

        #if isinstance(self.forward, Transform):
        #    if self.forward.input_array.shape == shape and self.axis == axis:
        #        # Already planned
        #        return

        plan_fwd = self._dst_fwd
        plan_bck = self._dst_bck

        # fftw wrapped with mpi4py-fft
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

        self.dst_fwd = xfftn_fwd
        self.dst_bck = xfftn_bck

    def sympy_basis(self, i=0, x=sp.Symbol('x', real=True)):
        f = sp.chebyshevu(i+2, x)/(i+3) - sp.chebyshevt(i+2, x)
        if self.is_scaled():
            return f
        return f*(i+2)*(i+3)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array[:] = eval_chebyu(i+2, x)/(i+3) - eval_chebyt(i+2, x)
        if not self.is_scaled():
            output_array *= (i+2)*(i+3)
        return output_array

    def evaluate_basis_all(self, x=None, argument=0):
        if x is None:
            x = self.mesh(False, False)
        T = Orthogonal.evaluate_basis_all(self, x=x, argument=argument)
        P = np.zeros_like(T)
        U = chebvanderU(x, self.shape(False)-1)
        k = np.arange(2, self.N)
        P[:, :-2] = U[:, 2:]/(k+1) - T[:, 2:]
        if not self.is_scaled():
            P[:, :-2] *= k*(k+1)
        if argument == 1: # if trial function
            P[:, -1] = (T[:, 0] + T[:, 1])/2    # x = +1
            P[:, -2] = (T[:, 0] - T[:, 1])/2    # x = -1
        return P

    def slice(self):
        return slice(0, self.N-2)

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        return SpectralBase.evaluate_basis_derivative(self, x=x, i=i, k=k, output_array=output_array)

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mesh(False, False)
        T = Orthogonal.evaluate_basis_derivative_all(self, x=x, k=k, argument=argument)
        V = n_cheb.chebvander(x, self.shape(False))
        M = V.shape[1]
        D = np.zeros((M, M))
        k = k+1
        D[slice(0, M-k)] = n_cheb.chebder(np.eye(M, M), k)
        V = np.dot(V, D)
        P = np.zeros_like(T)
        i = np.arange(self.N-2)[None, :]
        P[:, :-2] = V[:, 3:]/(i+3)**2 - T[:, 2:]
        if not self.is_scaled():
            P[:, :-2] *= (i+2)*(i+3)
        return P

    def _evaluate_expansion_all(self, input_array, output_array, x=None, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_expansion_all(self, input_array, output_array, x, False)
            return

        assert input_array is self.backward.tmp_array
        assert output_array is self.backward.output_array

        k = self.broadcast_to_ndims(np.arange(self.N))
        # First k*U_k (or just U_k)
        sN = self.si[self.N-1]
        se = self.sl[slice(0, self.N, 2)]
        so = self.sl[slice(1, self.N, 2)]
        s0 = self.sl[slice(0, self.N-2)]
        s2 = self.sl[slice(2, self.N)]

        self.dst_bck.input_array[s2] = input_array[s0]
        self.dst_bck.input_array[self.sl[slice(0, 2)]] = 0
        if self.is_scaled():
            self.dst_bck.input_array[s2] /= (k[s2]+1)

        out = self.dst_bck()
        out *= 0.5
        out[se] += input_array[sN]/2
        out[so] -= input_array[sN]/2
        out *= 1/(self.broadcast_to_ndims(np.sin((np.arange(self.N)+0.5)*np.pi/self.N)))

        # Then - T_k
        w0 = input_array.copy()
        w0[:] = 0
        w0[s2] = input_array[s0]
        #w0[s2] *=
        if not self.is_scaled():
            w0[s2] *= k[s2]*(k[s2]+1)
        self.bc.add_to_orthogonal(w0, input_array)
        input_array[:] = w0
        Orthogonal._evaluate_expansion_all(self, input_array, output_array, x, True)
        output_array *= -1
        output_array[:] += out

    def _evaluate_scalar_product(self, fast_transform=True):
        if fast_transform is False:
            SpectralBase._evaluate_scalar_product(self)
            self.scalar_product.output_array[self.si[-2]] = 0
            self.scalar_product.output_array[self.si[-1]] = 0
            return

        input_array = self.scalar_product._input_array
        output_array = self.scalar_product.output_array

        #self.dst_fwd.input_array[:] = input_array/self.broadcast_to_ndims(np.sin((np.arange(self.N)+0.5)*np.pi/self.N))
        #out = self.dst_fwd()
        #out *= (np.pi/(2*self.N*self.padding_factor))
        #k = self.broadcast_to_ndims(np.arange(2, self.N))
        #Orthogonal._evaluate_scalar_product(self, True)
        #s0 = self.sl[slice(0, self.N-2)]
        #s2 = self.sl[slice(2, self.N)]
        #if not self.is_scaled():
        #    output_array[s0] = -k*(k+1)*output_array[s2]
        #    output_array[s0] += k*out[s2]
        #else:
        #    output_array[s0] = -output_array[s2]
        #    output_array[s0] += out[s2]/(k+1)

        # It is also possible to do the scalar product using just one
        # Chebyshev transform.
        ft = np.zeros_like(input_array)
        input_array[:] = input_array/self.broadcast_to_ndims(np.sin((np.arange(self.N)+0.5)*np.pi/self.N)**2)
        Orthogonal._evaluate_scalar_product(self, True)
        output_array *= (2/np.pi)
        output_array[self.si[0]] /= 2
        ft = output_array.copy()
        fh = output_array
        s0 = self.sl[slice(0, self.N-2)]
        s2 = self.sl[slice(2, self.N)]
        s22 = self.sl[slice(2, self.N-2)]
        s02= self.sl[slice(0, self.N-4)]
        s4 = self.sl[slice(4, self.N)]
        k = self.broadcast_to_ndims(np.arange(self.N))
        fh[s0] /= (4*k[s2]*(k[s2]-1))
        fh[self.si[0]] *= 2
        fh[s0] -= ft[s2]/(2*(k[s2]+1)*(k[s2]-1))
        fh[s02] += ft[s4]/(4*k[s22]*(k[s22]+1))
        fh[s0] *= (k[s2]**2*(k[s2]+1)*(k[s2]-1)*np.pi/2)
        if self.is_scaled():
            fh[s0] /= (k[s2]*(k[s2]+1))

        output_array[self.si[-2]] = 0
        output_array[self.si[-1]] = 0

    @property
    def is_orthogonal(self):
        return False

    def get_bc_basis(self):
        if self._bc_basis:
            return self._bc_basis
        self._bc_basis = BCDirichlet(self.N, quad=self.quad, domain=self.domain,
                                     coordinates=self.coors.coordinates)
        return self._bc_basis

    def get_refined(self, N):
        return MixedTU(N,
                       quad=self.quad,
                       domain=self.domain,
                       dtype=self.dtype,
                       padding_factor=self.padding_factor,
                       dealias_direct=self.dealias_direct,
                       coordinates=self.coors.coordinates,
                       bc=self.bc.bc,
                       scaled=self._scaled)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False):
        return MixedTU(self.N,
                       quad=self.quad,
                       domain=self.domain,
                       dtype=self.dtype,
                       padding_factor=padding_factor,
                       dealias_direct=dealias_direct,
                       coordinates=self.coors.coordinates,
                       bc=self.bc.bc,
                       scaled=self._scaled)

    def get_unplanned(self):
        return MixedTU(self.N,
                       quad=self.quad,
                       domain=self.domain,
                       dtype=self.dtype,
                       padding_factor=self.padding_factor,
                       dealias_direct=self.dealias_direct,
                       coordinates=self.coors.coordinates,
                       bc=self.bc.bc,
                       scaled=self._scaled)


class BCDirichlet(CompositeSpace):
    """Function space for Dirichlet boundary conditions

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

    def slice(self):
        return slice(self.N-2, self.N)

    def shape(self, forward_output=True):
        if forward_output:
            return 2
        else:
            return self.N

    @staticmethod
    def boundary_condition():
        return 'Apply'

    @staticmethod
    def short_name():
        return 'BCD'

    def vandermonde(self, x):
        return n_cheb.chebvander(x, 1)

    def coefficient_matrix(self):
        return np.array([[0.5, -0.5],
                         [0.5, 0.5]])

    def _composite(self, V, argument=0):
        P = np.zeros(V[:, :2].shape)
        #P[:, 0] = (V[:, 0] - V[:, 1])/2
        #P[:, 1] = (V[:, 0] + V[:, 1])/2
        P[:] = np.tensordot(V[:, :2], self.coefficient_matrix(), (1, 1))
        return P

    def sympy_basis(self, i=0, x=xp):
        if i == 0:
            return 0.5*(1-x)
        elif i == 1:
            return 0.5*(1+x)
        else:
            raise AttributeError('Only two bases, i < 2')

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0:
            output_array[:] = 0.5*(1-x)
        elif i == 1:
            output_array[:] = 0.5*(1+x)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0 and k == 1:
            output_array[:] = -0.5
        elif i == 1 and k == 1:
            output_array[:] = 0.5
        else:
            output_array[:] = 0
        return output_array

class BCNeumann(CompositeSpace):

    def __init__(self, N, quad="GC", scaled=False, dtype=float,
                 domain=(-1., 1.), coordinates=None, **kw):
        CompositeSpace.__init__(self, N, quad=quad, dtype=dtype, domain=domain, coordinates=coordinates)
        self._scaled = scaled

    def slice(self):
        return slice(self.N-2, self.N)

    def shape(self, forward_output=True):
        if forward_output:
            return 2
        else:
            return self.N

    @staticmethod
    def boundary_condition():
        return 'Apply'

    @staticmethod
    def short_name():
        return 'BCN'

    def vandermonde(self, x):
        return n_cheb.chebvander(x, 2)

    @staticmethod
    def coefficient_matrix():
        return np.array([[0, 1/2, -1/8],
                         [0, 1/2, 1/8]])

    def _composite(self, V, argument=0):
        P = np.zeros(V[:, :2].shape)
        P[:, 0] = 0.5*V[:, 1] - 1/8*V[:, 2]
        P[:, 1] = 0.5*V[:, 1] + 1/8*V[:, 2]
        return P

    def sympy_basis(self, i=0, x=sp.symbols('x', real=True)):
        if i == 0:
            return x/2-(2*x**2-1)/8
        elif i == 1:
            return x/2+(2*x**2-1)/8
        else:
            raise AttributeError('Only two bases, i < 2')

    def evaluate_basis(self, x, i=0, output_array=None):
        assert i < 2
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0:
            output_array[:] = x/2-(2*x**2-1)/8
        elif i == 1:
            output_array[:] = x/2+(2*x**2-1)/8
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0 and k == 0:
            output_array[:] = x/2-(2*x**2-1)/8
        elif i == 0 and k == 1:
            output_array[:] = 0.5-0.5*x
        elif i == 0 and k == 2:
            output_array[:] = -0.5
        elif i == 1 and k == 0:
            output_array[:] = x/2+(2*x**2-1)/8
        elif i == 1 and k == 1:
            output_array[:] = 0.5+0.5*x
        elif i == 1 and k == 2:
            output_array[:] = 0.5
        else:
            output_array[:] = 0
        return output_array


class BCBiharmonic(CompositeSpace):
    """Function space for inhomogeneous Biharmonic boundary conditions

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
        CompositeSpace.__init__(self, N, quad=quad, dtype=dtype, domain=domain, coordinates=coordinates)

    def slice(self):
        return slice(self.N-4, self.N)

    def shape(self, forward_output=True):
        if forward_output:
            return 4
        else:
            return self.N

    @staticmethod
    def boundary_condition():
        return 'Apply'

    @staticmethod
    def short_name():
        return 'BCB'

    def vandermonde(self, x):
        return n_cheb.chebvander(x, 3)

    @staticmethod
    def coefficient_matrix():
        return np.array([[0.5, -9./16., 0, 1./16],
                         [0.5, 9./16., 0, -1./16.],
                         [1./8., -1./16., -1./8., 1./16.],
                         [-1./8., -1./16., 1./8., 1./16.]])

    def _composite(self, V, argument=0):
        P = np.tensordot(V[:, :4], self.coefficient_matrix(), (1, 1))
        return P

    def sympy_basis(self, i=0, x=xp):
        assert i < 4, 'Only four bases, i < 4'
        return np.sum(self.coefficient_matrix()[i]*np.array([sp.chebyshevt(j, x) for j in range(4)]))

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        V = self.vandermonde(x)
        output_array[:] = np.dot(V, self.coefficient_matrix()[i])
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        output_array = SpectralBase.evaluate_basis_derivative(self, x=x, i=i, k=k, output_array=output_array)
        return output_array

class BCUpperDirichlet(CompositeSpace):
    """Function space for Dirichlet boundary conditions at x=1

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
        CompositeSpace.__init__(self, N, quad=quad, dtype=dtype, domain=domain,
                                coordinates=coordinates)

    def slice(self):
        return slice(self.N-1, self.N)

    def shape(self, forward_output=True):
        if forward_output:
            return 1
        else:
            return self.N

    @staticmethod
    def boundary_condition():
        return 'Apply'

    @staticmethod
    def short_name():
        return 'BCUD'

    def vandermonde(self, x):
        return n_cheb.chebvander(x, 1)

    def coefficient_matrix(self):
        return np.array([[0.5, 0.5]])

    def _composite(self, V, argument=0):
        P = np.zeros(V[:, :1].shape)
        P[:, 0] = (V[:, 0] + V[:, 1])/2
        return P

    def sympy_basis(self, i=0, x=xp):
        if i == 0:
            return 0.5*(1+x)
        else:
            raise AttributeError('Only one basis, i == 0')

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0:
            output_array[:] = 0.5*(1+x)
        else:
            raise AttributeError('Only one basis, i == 0')
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        assert i == 0
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        output_array[:] = 0
        if k == 1:
            output_array[:] = 0.5
        elif k == 0:
            output_array[:] = 0.5*(1+x)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mesh(False, False)
        output_array = np.zeros((x.shape[0], 1))
        self.evaluate_basis_derivative(x=x, k=k, output_array=output_array[:, 0])
        return output_array

class BCNeumannDirichlet(CompositeSpace):

    def __init__(self, N, quad="GC", dtype=float,
                 domain=(-1., 1.), coordinates=None, **kw):
        CompositeSpace.__init__(self, N, quad=quad, dtype=dtype, domain=domain, coordinates=coordinates)

    def slice(self):
        return slice(self.N-2, self.N)

    def shape(self, forward_output=True):
        if forward_output:
            return 2
        else:
            return self.N

    @staticmethod
    def boundary_condition():
        return 'Apply'

    @staticmethod
    def short_name():
        return 'BCND'

    def vandermonde(self, x):
        return n_cheb.chebvander(x, 2)

    @staticmethod
    def coefficient_matrix():
        return np.array([[1, -0.6, -0.4],
                         [1, 0, 0]])

    def _composite(self, V, argument=0):
        P = np.zeros(V[:, :2].shape)
        P[:, 0] = V[:, 0] - 0.6*V[:, 1] - 0.4*V[:, 2]
        P[:, 1] = V[:, 0]
        return P

    def sympy_basis(self, i=0, x=sp.symbols('x', real=True)):
        if i == 0:
            return 1-0.6*x-0.4*(2*x**2-1)
        elif i == 1:
            return 1
        else:
            raise AttributeError('Only two bases, i < 2')

    def evaluate_basis(self, x, i=0, output_array=None):
        assert i < 2
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0:
            output_array[:] = 1-0.6*x-0.4*(2*x**2-1)
        elif i == 1:
            output_array[:] = 1
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0 and k == 0:
            output_array[:] = 1-0.6*x-0.4*(2*x**2-1)
        elif i == 0 and k == 1:
            output_array[:] = -0.6-1.6*x
        elif i == 0 and k == 2:
            output_array[:] = -1.6
        elif i == 1 and k == 0:
            output_array[:] = 1
        else:
            output_array[:] = 0
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mesh(False, False)
        output_array = np.zeros((self.N, 2))
        self.evaluate_basis_derivative(x=x, i=0, k=k, output_array=output_array[:, 0])
        self.evaluate_basis_derivative(x=x, i=1, k=k, output_array=output_array[:, 1])
        return output_array

class BCDirichletNeumann(CompositeSpace):

    def __init__(self, N, quad="GC", dtype=float,
                 domain=(-1., 1.), coordinates=None, **kw):
        CompositeSpace.__init__(self, N, quad=quad, dtype=dtype, domain=domain, coordinates=coordinates)

    def slice(self):
        return slice(self.N-2, self.N)

    def shape(self, forward_output=True):
        if forward_output:
            return 2
        else:
            return self.N

    @staticmethod
    def boundary_condition():
        return 'Apply'

    @staticmethod
    def short_name():
        return 'BCDN'

    def vandermonde(self, x):
        return n_cheb.chebvander(x, 1)

    @staticmethod
    def coefficient_matrix():
        return np.array([[1, 0],
                         [1, 1]])

    def _composite(self, V, argument=0):
        P = np.zeros(V[:, :2].shape)
        P[:, 0] = V[:, 0]
        P[:, 1] = V[:, 0] + V[:, 1]
        return P

    def sympy_basis(self, i=0, x=sp.symbols('x', real=True)):
        if i == 0:
            return 1
        elif i == 1:
            return 1+x
        else:
            raise AttributeError('Only two bases, i < 2')

    def evaluate_basis(self, x, i=0, output_array=None):
        assert i < 2
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0:
            output_array[:] = 1
        elif i == 1:
            output_array[:] = 1+x
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0 and k == 0:
            output_array[:] = 1
        elif i == 1 and k == 0:
            output_array[:] = 1+x
        elif i == 1 and k == 1:
            output_array[:] = 1
        else:
            output_array[:] = 0
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mesh(False, False)
        output_array = np.zeros((x.shape[0], 2))
        self.evaluate_basis_derivative(x=x, i=0, k=k, output_array=output_array[:, 0])
        self.evaluate_basis_derivative(x=x, i=1, k=k, output_array=output_array[:, 1])
        return output_array

class BCUpperDirichletNeumann(CompositeSpace):

    def __init__(self, N, quad="GC", dtype=float,
                 domain=(-1., 1.), coordinates=None, **kw):
        CompositeSpace.__init__(self, N, quad=quad, dtype=dtype, domain=domain, coordinates=coordinates)

    def slice(self):
        return slice(self.N-2, self.N)

    def shape(self, forward_output=True):
        if forward_output:
            return 2
        else:
            return self.N

    @staticmethod
    def boundary_condition():
        return 'Apply'

    @staticmethod
    def short_name():
        return 'BCUDN'

    def vandermonde(self, x):
        return n_cheb.chebvander(x, 2)

    @staticmethod
    def coefficient_matrix():
        return np.array([[1, 0, 0],
                         [1, -5/3, 2/3]])

    def _composite(self, V, argument=0):
        P = np.zeros(V[:, :2].shape)
        P[:, 0] = V[:, 0]
        P[:, 1] = V[:, 0]-5/3*V[:, 1]+2/3*V[:, 2]
        return P

    def sympy_basis(self, i=0, x=sp.symbols('x', real=True)):
        if i == 0:
            return 1
        elif i == 1:
            return 1-5/3*x+2/3*(2*x**2-1)
        else:
            raise AttributeError('Only two bases, i < 2')

    def evaluate_basis(self, x, i=0, output_array=None):
        assert i < 2
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0:
            output_array[:] = 1
        elif i == 1:
            output_array[:] = 1-5/3*x+2/3*(2*x**2-1)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i == 0 and k == 0:
            output_array[:] = 1
        elif i == 1 and k == 0:
            output_array[:] = 1-5/3*x+2/3*(2*x**2-1)
        elif i == 1 and k == 1:
            output_array[:] = -5/3+8/3*x
        elif i == 1 and k == 2:
            output_array[:] = 8/3
        else:
            output_array[:] = 0
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0, argument=0):
        if x is None:
            x = self.mesh(False, False)
        output_array = np.zeros((x.shape[0], 2))
        self.evaluate_basis_derivative(x=x, i=0, k=k, output_array=output_array[:, 0])
        self.evaluate_basis_derivative(x=x, i=1, k=k, output_array=output_array[:, 1])
        return output_array

def chebvanderU(x, deg):
    """Pseudo-Vandermonde matrix of given degree for Chebyshev polynomials
    of second kind.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = T_i(x),

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
