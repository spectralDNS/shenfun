"""
Module for defining bases in the Chebyshev family
"""
import functools
import numpy as np
from numpy.polynomial import chebyshev as n_cheb
from scipy.special import eval_chebyt
import pyfftw
from mpi4py_fft import fftw
from shenfun.spectralbase import SpectralBase, work, Transform, FuncWrap
from shenfun.optimization import Cheb
from shenfun.utilities import inheritdocstrings

__all__ = ['ChebyshevBase', 'Basis', 'ShenDirichletBasis',
           'ShenNeumannBasis', 'ShenBiharmonicBasis']

#pylint: disable=abstract-method, not-callable, method-hidden, no-self-use, cyclic-import

class DCTWrap(FuncWrap):

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


@inheritdocstrings
class ChebyshevBase(SpectralBase):
    """Abstract base class for all Chebyshev bases

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
    """

    def __init__(self, N=0, quad="GC", domain=(-1., 1.)):
        assert quad in ('GC', 'GL')
        SpectralBase.__init__(self, N, quad, domain=domain)

    @staticmethod
    def family():
        return 'chebyshev'

    def points_and_weights(self, N=None, map_true_domain=False):
        if N is None:
            N = self.N
        if self.quad == "GL":
            points = -(n_cheb.chebpts2(N)).astype(float)
            weights = np.full(N, np.pi/(N-1))
            weights[0] /= 2
            weights[-1] /= 2

        elif self.quad == "GC":
            points, weights = n_cheb.chebgauss(N)
            points = points.astype(float)
            weights = weights.astype(float)

        if map_true_domain is True:
            points = self.map_true_domain(points)

        return points, weights

    def vandermonde(self, x):
        return n_cheb.chebvander(x, self.N-1)

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        #output_array[:] = np.cos(i*np.arccos(x))
        output_array[:] = eval_chebyt(i, x)
        return output_array

    def evaluate_basis_derivative_all(self, x=None, k=0):
        if x is None:
            x = self.mesh(False, False)
        V = self.vandermonde(x)
        if k > 0:
            D = np.zeros((self.N, self.N))
            D[:-k, :] = n_cheb.chebder(np.eye(self.N), k)
            V = np.dot(V, D)
        return self._composite_basis(V)

    def evaluate_basis_all(self, x=None):
        if x is None:
            x = self.mesh(False, False)
        V = self.vandermonde(x)
        return self._composite_basis(V)

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        x = np.atleast_1d(x)
        v = self.evaluate_basis(x, i, output_array)
        if k > 0:
            D = np.zeros((self.N, self.N))
            D[:-k, :] = n_cheb.chebder(np.eye(self.N), k)
            v = np.dot(v, D)
        return v

    def _composite_basis(self, V):
        """Return composite basis, where ``V`` is primary Vandermonde matrix."""
        return V

    def reference_domain(self):
        return (-1., 1.)

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[-1]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        plan_fwd = self._xfftn_fwd
        plan_bck = self._xfftn_bck

        if 'builders' in self._xfftn_fwd.func.__module__:
            opts = dict(
                avoid_copy=True,
                overwrite_input=True,
                auto_align_input=True,
                auto_contiguous=True,
                planner_effort='FFTW_MEASURE',
                threads=1,
            )
            opts.update(options)

            U = pyfftw.empty_aligned(shape, dtype=np.float)
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

            U = pyfftw.empty_aligned(shape, dtype=np.float)

            xfftn_fwd = plan_fwd(U, axes=(axis,), threads=threads, flags=flags)
            V = xfftn_fwd.output_array
            xfftn_bck = plan_bck(V, axes=(axis,), threads=threads, flags=flags, output_array=U)
            V.fill(0)
            U.fill(0)

        if np.dtype(dtype) is np.dtype('complex'):
            # dct only works on real data, so need to wrap it
            U = pyfftw.empty_aligned(shape, dtype=np.complex)
            V = pyfftw.empty_aligned(shape, dtype=np.complex)
            U.fill(0)
            V.fill(0)
            xfftn_fwd = DCTWrap(xfftn_fwd, U, V)
            xfftn_bck = DCTWrap(xfftn_bck, V, U)

        self._ndim_tensor = U.ndim
        self.axis = axis
        self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
        self.backward = Transform(self.backward, xfftn_bck, V, V, U)
        self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)


@inheritdocstrings
class Basis(ChebyshevBase):
    """Basis for regular Chebyshev series

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
    """

    def __init__(self, N=0, quad="GC", domain=(-1., 1.)):
        ChebyshevBase.__init__(self, N, quad, domain)
        if quad == 'GC':
            self._xfftn_fwd = functools.partial(fftw.dctn, type=2)
            self._xfftn_bck = functools.partial(fftw.dctn, type=3)

        else:
            self._xfftn_fwd = functools.partial(fftw.dctn, type=1)
            self._xfftn_bck = functools.partial(fftw.dctn, type=1)
        self.plan(N, 0, np.float, {})

    @staticmethod
    def derivative_coefficients(fk):
        """Return coefficients of Chebyshev series for c = f'(x)

        Parameters
        ----------
            fk : input array
                 Coefficients of regular Chebyshev series

        Returns
        -------
        array
            Coefficients of derivative of fk-series

        """
        ck = np.zeros_like(fk)
        if len(fk.shape) == 1:
            ck = Cheb.derivative_coefficients(fk, ck)
        elif len(fk.shape) == 3:
            ck = Cheb.derivative_coefficients_3D(fk, ck)
        return ck

    def fast_derivative(self, fj):
        """Return derivative of :math:`f_j = f(x_j)` at quadrature points
        :math:`x_j`

        Parameters
        ----------
            fj : input array
                 Function values on quadrature mesh

        Returns
        -------
        array
             Array with derivatives on quadrature mesh
        """
        fk = self.forward(fj)
        ck = self.derivative_coefficients(fk)
        fd = self.backward(ck)
        return fd.copy()

    def apply_inverse_mass(self, array):
        sl = self.sl
        array *= (2/np.pi)
        array[sl(0)] /= 2
        if self.quad == 'GL':
            array[sl(-1)] /= 2
        return array

    def evaluate_expansion_all(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            SpectralBase.evaluate_expansion_all(self, input_array, output_array, False)
            return

        output_array = self.backward.xfftn()
        sl = self.sl
        if self.quad == "GC":
            output_array *= 0.5
            output_array += input_array[sl(slice(0, 1))]/2

        elif self.quad == "GL":
            output_array *= 0.5
            output_array += input_array[sl(slice(0, 1))]/2
            s0 = sl(slice(-1, None))
            s2 = sl(slice(0, None, 2))
            output_array[s2] += input_array[s0]/2
            s2 = sl(slice(1, None, 2))
            output_array[s2] -= input_array[s0]/2

    def evaluate_scalar_product(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            self.vandermonde_scalar_product(input_array, output_array)
            return
        assert self.N == self.scalar_product.input_array.shape[self.axis]
        out = self.scalar_product.xfftn()
        if self.quad == "GC":
            out *= (np.pi/(2*self.N))

        elif self.quad == "GL":
            out *= (np.pi/(2*(self.N-1)))

    def eval(self, x, u, output_array=None):
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = self.map_reference_domain(x)
        output_array[:] = n_cheb.chebval(x, u)
        return output_array


@inheritdocstrings
class ShenDirichletBasis(ChebyshevBase):
    """Shen basis for Dirichlet boundary conditions

    Parameters
    ----------
        N : int, optional
            Number of quadrature points
        quad : str, optional
               Type of quadrature

               - GL - Chebyshev-Gauss-Lobatto
               - GC - Chebyshev-Gauss

        bc : 2-tuple of floats, optional
             Boundary conditions at x=(1,-1). For Poisson eq.
        domain : 2-tuple of floats, optional
                 The computational domain
        scaled : bool, optional
                 Whether or not to use scaled basis
    """

    def __init__(self, N=0, quad="GC", bc=(0, 0),
                 domain=(-1., 1.), scaled=False):
        ChebyshevBase.__init__(self, N, quad, domain=domain)
        from shenfun.tensorproductspace import BoundaryValues
        self.CT = Basis(N, quad)
        self._scaled = scaled
        self._factor = np.ones(1)
        self.plan(N, 0, np.float, {})
        self.bc = BoundaryValues(self, bc=bc)

    @staticmethod
    def boundary_condition():
        return 'Dirichlet'

    def _composite_basis(self, V):
        P = np.zeros(V.shape)
        P[:, :-2] = V[:, :-2] - V[:, 2:]
        P[:, -2] = (V[:, 0] + V[:, 1])/2
        P[:, -1] = (V[:, 0] - V[:, 1])/2
        return P

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        if i < self.N-2:
            w = np.arccos(x)
            output_array[:] = np.cos(i*w) - np.cos((i+2)*w)
        elif i == self.N-2:
            output_array[:] = 0.5*(1+x)
        elif i == self.N-1:
            output_array[:] = 0.5*(1-x)
        return output_array

    def is_scaled(self):
        """Return True if scaled basis is used, otherwise False"""
        return False

    def evaluate_scalar_product(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            self.vandermonde_scalar_product(input_array, output_array)
            return
        output = self.CT.scalar_product(fast_transform=fast_transform)
        s0 = self.sl(0)
        s1 = self.sl(1)
        c0 = 0.5*(output[s0] + output[s1])
        c1 = 0.5*(output[s0] - output[s1])
        s0 = self.sl(slice(0, -2))
        s1 = self.sl(slice(2, None))
        output[s0] -= output[s1]
        s0 = list(s0)
        s0[self.axis] = -2
        output[tuple(s0)] = c0
        s0[self.axis] = -1
        output[tuple(s0)] = c1

    def evaluate_expansion_all(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            SpectralBase.evaluate_expansion_all(self, input_array, output_array, False)
            return
        w_hat = work[(input_array, 0)]
        s0 = self.sl(slice(0, -2))
        s1 = self.sl(slice(2, None))
        w_hat[s0] = input_array[s0]
        w_hat[s1] -= input_array[s0]
        self.bc.apply_before(w_hat, False, (0.5, 0.5))
        self.CT.backward(w_hat)
        assert output_array is self.CT.backward.output_array

    def slice(self):
        return slice(0, self.N-2)

    def eval(self, x, u, output_array=None):
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0)]
        output_array[:] = n_cheb.chebval(x, u[:-2])
        w_hat[2:] = u[:-2]
        output_array -= n_cheb.chebval(x, w_hat)
        output_array += 0.5*(u[-1]*(1-x)+u[-2]*(1+x))
        return output_array

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[-1]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        self.CT.plan(shape, axis, dtype, options)
        xfftn_fwd = self.CT.forward.xfftn
        xfftn_bck = self.CT.backward.xfftn
        U = xfftn_fwd.input_array
        V = xfftn_fwd.output_array
        self._ndim_tensor = U.ndim
        self.axis = axis
        self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
        self.backward = Transform(self.backward, xfftn_bck, V, V, U)
        self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)


@inheritdocstrings
class ShenNeumannBasis(ChebyshevBase):
    """Shen basis for homogeneous Neumann boundary conditions

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
    """

    def __init__(self, N=0, quad="GC", mean=0, domain=(-1., 1.)):
        ChebyshevBase.__init__(self, N, quad, domain=domain)
        self.mean = mean
        self.CT = Basis(N, quad)
        self._factor = np.zeros(0)
        self.plan(N, 0, np.float, {})

    @staticmethod
    def boundary_condition():
        return 'Neumann'

    def _composite_basis(self, V):
        assert self.N == V.shape[1]
        P = np.zeros(V.shape)
        k = np.arange(self.N).astype(np.float)
        P[:, :-2] = V[:, :-2] - (k[:-2]/(k[:-2]+2))**2*V[:, 2:]
        return P

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        w = np.arccos(x)
        output_array[:] = np.cos(i*w) - (i*1./(i+2))**2*np.cos((i+2)*w)
        return output_array

    def set_factor_array(self, v):
        """Set intermediate factor arrays"""
        if not self._factor.shape == v.shape:
            k = self.wavenumbers().astype(float)
            self._factor = (k/(k+2))**2

    def evaluate_scalar_product(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            self.vandermonde_scalar_product(input_array, output_array)
            return
        output = self.CT.scalar_product(fast_transform=True)
        self.set_factor_array(output)
        sm2 = self.sl(slice(0, -2))
        s2p = self.sl(slice(2, None))
        output[sm2] -= self._factor * output[s2p]

    def scalar_product(self, input_array=None, output_array=None, fast_transform=True):
        if input_array is not None:
            self.scalar_product.input_array[...] = input_array

        self.evaluate_scalar_product(self.scalar_product.input_array,
                                     self.scalar_product.output_array,
                                     fast_transform=fast_transform)

        output = self.scalar_product.output_array
        output[self.sl(0)] = self.mean*np.pi
        output[self.sl(slice(-2, None))] = 0

        if output_array is not None:
            output_array[...] = output
            return output_array
        return output

    def evaluate_expansion_all(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            SpectralBase.evaluate_expansion_all(self, input_array, output_array, False)
            return
        w_hat = work[(input_array, 0)]
        self.set_factor_array(input_array)
        s0 = self.sl(slice(0, -2))
        s1 = self.sl(slice(2, None))
        w_hat[s0] = input_array[s0]
        w_hat[s1] -= self._factor*input_array[s0]
        self.CT.backward(w_hat)
        assert output_array is self.CT.backward.output_array

    def slice(self):
        return slice(0, self.N-2)

    def eval(self, x, u, output_array=None):
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0)]
        self.set_factor_array(u)
        output_array[:] = n_cheb.chebval(x, u[:-2])
        w_hat[2:] = self._factor*u[:-2]
        output_array -= n_cheb.chebval(x, w_hat)
        return output_array

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[0]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        self.CT.plan(shape, axis, dtype, options)
        xfftn_fwd = self.CT.forward.xfftn
        xfftn_bck = self.CT.backward.xfftn
        U = xfftn_fwd.input_array
        V = xfftn_fwd.output_array
        self._ndim_tensor = U.ndim
        self.axis = axis
        self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
        self.backward = Transform(self.backward, xfftn_bck, V, V, U)
        self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)


@inheritdocstrings
class ShenBiharmonicBasis(ChebyshevBase):
    """Shen biharmonic basis

    Homogeneous Dirichlet and Neumann boundary conditions.

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

    """

    def __init__(self, N=0, quad="GC", domain=(-1., 1.)):
        ChebyshevBase.__init__(self, N, quad, domain=domain)
        self.CT = Basis(N, quad)
        self._factor1 = np.zeros(0)
        self._factor2 = np.zeros(0)
        self.plan(N, 0, np.float, {})

    @staticmethod
    def boundary_condition():
        return 'Biharmonic'

    def _composite_basis(self, V):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(np.float)[:-4]
        P[:, :-4] = V[:, :-4] - (2*(k+2)/(k+3))*V[:, 2:-2] + ((k+1)/(k+3))*V[:, 4:]
        return P

    def evaluate_basis(self, x, i=0, output_array=None):
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape)
        w = np.arccos(x)
        output_array[:] = np.cos(i*w) - (2*(i+2.)/(i+3.))*np.cos((i+2)*w) + ((i+1.)/(i+3.))*np.cos((i+4)*w)
        return output_array

    def set_factor_arrays(self, v):
        """Set intermediate factor arrays"""
        s = self.sl(self.slice())
        if not self._factor1.shape == v[s].shape:
            k = self.wavenumbers().astype(float)
            self._factor1 = (-2*(k+2)/(k+3)).astype(float)
            self._factor2 = ((k+1)/(k+3)).astype(float)

    def evaluate_scalar_product(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            self.vandermonde_scalar_product(input_array, output_array)
            return
        output = self.CT.scalar_product(fast_transform=fast_transform)
        Tk = work[(output, 0)]
        Tk[...] = output
        self.set_factor_arrays(Tk)
        s = self.sl(self.slice())
        s2 = self.sl(slice(2, -2))
        s4 = self.sl(slice(4, None))
        output[s] += self._factor1 * Tk[s2]
        output[s] += self._factor2 * Tk[s4]

    def scalar_product(self, input_array=None, output_array=None, fast_transform=True):
        if input_array is not None:
            self.scalar_product.input_array[...] = input_array

        self.evaluate_scalar_product(self.scalar_product.input_array,
                                     self.scalar_product.output_array,
                                     fast_transform=fast_transform)

        self.scalar_product.output_array[self.sl(slice(-4, None))] = 0

        if output_array is not None:
            output_array[...] = self.scalar_product.output_array
            return output_array
        return self.scalar_product.output_array

    #@optimizer
    def set_w_hat(self, w_hat, fk, f1, f2):
        """Return intermediate w_hat array"""
        s = self.sl(self.slice())
        s2 = self.sl(slice(2, -2))
        s4 = self.sl(slice(4, None))
        w_hat[s] = fk[s]
        w_hat[s2] += f1*fk[s]
        w_hat[s4] += f2*fk[s]
        return w_hat

    def evaluate_expansion_all(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            SpectralBase.evaluate_expansion_all(self, input_array, output_array, False)
            return
        w_hat = work[(input_array, 0)]
        self.set_factor_arrays(input_array)
        w_hat = self.set_w_hat(w_hat, input_array, self._factor1, self._factor2)
        self.CT.backward(w_hat)
        assert input_array is self.backward.input_array
        assert output_array is self.backward.output_array

    def slice(self):
        return slice(0, self.N-4)

    def eval(self, x, u, output_array=None):
        if output_array is None:
            output_array = np.zeros(x.shape)
        x = self.map_reference_domain(x)
        w_hat = work[(u, 0)]
        self.set_factor_arrays(u)
        output_array[:] = n_cheb.chebval(x, u[:-4])
        w_hat[2:-2] = self._factor1*u[:-4]
        output_array += n_cheb.chebval(x, w_hat[:-2])
        w_hat[4:] = self._factor2*u[:-4]
        w_hat[:4] = 0
        output_array += n_cheb.chebval(x, w_hat)
        return output_array

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, tuple):
            assert len(axis) == 1
            axis = axis[-1]

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and self.axis == axis:
                # Already planned
                return

        self.CT.plan(shape, axis, dtype, options)
        xfftn_fwd = self.CT.forward.xfftn
        xfftn_bck = self.CT.backward.xfftn
        U = xfftn_fwd.input_array
        V = xfftn_fwd.output_array
        self._ndim_tensor = U.ndim
        self.axis = axis
        self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
        self.backward = Transform(self.backward, xfftn_bck, V, V, U)
        self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)
