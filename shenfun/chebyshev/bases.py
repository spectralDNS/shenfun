from numpy.polynomial import chebyshev as n_cheb
import functools
import numpy as np
import pyfftw
from mpiFFT4py import dct
from shenfun.spectralbase import SpectralBase, work, _func_wrap
from shenfun.optimization import Cheb
from shenfun.utilities import inheritdocstrings

__all__ = ['ChebyshevBase', 'Basis', 'ShenDirichletBasis',
           'ShenNeumannBasis', 'ShenBiharmonicBasis']


class _dct_wrap(object):

    # pylint: disable=too-few-public-methods

    __slots__ = ('_dct', 'input_array', 'output_array')

    def __init__(self, xfftn, in_array, out_array):
        object.__setattr__(self, '_dct', xfftn)
        object.__setattr__(self, 'input_array', in_array)
        object.__setattr__(self, 'output_array', out_array)

    def __getattribute__(self, name):
        if name in ('input_array', 'output_array'):
            return object.__getattribute__(self, name)
        else:
            dct_obj = object.__getattribute__(self, '_dct')
            return getattr(dct_obj, name)

    def __call__(self, input_array=None, output_array=None, **kw):
        dct_obj = object.__getattribute__(self, '_dct')

        if input_array is not None:
            self.input_array[...] = input_array

        dct_obj.input_array[...] = self.input_array.real
        dct_obj(None, None, **kw)
        self.output_array.real[:] = dct_obj.output_array
        dct_obj.input_array[...] = self.input_array.imag
        dct_obj(None, None, **kw)
        self.output_array.imag[:] = dct_obj.output_array

        if output_array is not None:
            output_array[...] = self.output_array
            return output_array
        else:
            return self.output_array


@inheritdocstrings
class ChebyshevBase(SpectralBase):
    """Abstract base class for all Chebyshev bases

    kwargs:
        N             int         Number of quadrature points
        quad        ('GL', 'GC')  Chebyshev-Gauss-Lobatto or Chebyshev-Gauss

    """

    def __init__(self, N=0, quad="GC"):
        assert quad in ('GC', 'GL')
        SpectralBase.__init__(self, N, quad)

    def points_and_weights(self):
        if self.quad == "GL":
            points = -(n_cheb.chebpts2(self.N)).astype(float)
            weights = np.zeros(self.N)+np.pi/(self.N-1)
            weights[0] /= 2
            weights[-1] /= 2

        elif self.quad == "GC":
            points, weights = n_cheb.chebgauss(self.N)
            points = points.astype(float)
            weights = weights.astype(float)

        return points, weights

    def vandermonde(self, x):
        """Return Chebyshev Vandermonde matrix

        args:
            x               points for evaluation

        """
        return n_cheb.chebvander(x, self.N-1)

    def get_vandermonde_basis_derivative(self, V, k=0):
        """Return k'th derivative of basis as a Vandermonde matrix

        args:
            V               Chebyshev Vandermonde matrix

        kwargs:
            k    integer    k'th derivative

        """
        assert self.N == V.shape[1]
        if k > 0:
            D = np.zeros((self.N, self.N))
            D[:-k, :] = n_cheb.chebder(np.eye(self.N), k)
            V = np.dot(V, D)
        return self.get_vandermonde_basis(V)

    def get_mass_matrix(self):
        from .matrices import mat
        return mat[((self.__class__, 0), (self.__class__, 0))]

    def plan(self, shape, axis, dtype, options):
        opts = dict(
            avoid_copy=True,
            overwrite_input=True,
            auto_align_input=True,
            auto_contiguous=True,
            planner_effort='FFTW_MEASURE',
            threads=1,
        )
        opts.update(options)

        plan_fwd = self._xfftn_fwd
        plan_bck = self._xfftn_bck
        if isinstance(axis, tuple):
            axis = axis[0]

        U = pyfftw.empty_aligned(shape, dtype=np.float)
        xfftn_fwd = plan_fwd(U, axis=axis, **opts)
        U.fill(0)
        V = xfftn_fwd.output_array
        xfftn_bck = plan_bck(V, axis=axis, **opts)
        V.fill(0)

        xfftn_fwd.update_arrays(U, V)
        xfftn_bck.update_arrays(V, U)

        self.axis = axis
        if np.dtype(dtype) is np.dtype('float64'):
            self.xfftn_fwd = xfftn_fwd
            self.xfftn_bck = xfftn_bck

        else:
            # dct only works on real data, so need to wrap it
            U = pyfftw.empty_aligned(shape, dtype=np.complex)
            V = pyfftw.empty_aligned(shape, dtype=np.complex)
            U.fill(0)
            V.fill(0)
            self.xfftn_fwd = _dct_wrap(xfftn_fwd, U, V)
            self.xfftn_bck = _dct_wrap(xfftn_bck, V, U)

        self.forward = _func_wrap(self.forward, self.xfftn_fwd)
        self.backward = _func_wrap(self.backward, self.xfftn_bck)
        self.scalar_product = _func_wrap(self.scalar_product, self.xfftn_fwd)


@inheritdocstrings
class Basis(ChebyshevBase):
    """Basis for regular Chebyshev series

    kwargs:
        N             int         Number of quadrature points
        quad        ('GL', 'GC')  Chebyshev-Gauss-Lobatto or Chebyshev-Gauss
        plan         boolean      Execute plan assuming 1D

    """

    def __init__(self, N=0, quad="GC", plan=False):
        ChebyshevBase.__init__(self, N, quad)
        if quad == 'GC':
            self._xfftn_fwd = functools.partial(pyfftw.builders.dct, type=2)
            self._xfftn_bck = functools.partial(pyfftw.builders.dct, type=3)
        else:
            self._xfftn_fwd = functools.partial(pyfftw.builders.dct, type=1)
            self._xfftn_bck = functools.partial(pyfftw.builders.dct, type=1)
        if plan:
            self.plan(N, 0, np.float, {})


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

    def apply_inverse_mass(self, array):
        """Apply inverse BTT_{kj} = c_k 2/pi \delta_{kj}

        args:
            array   (input/output)    Expansion coefficients

        """
        sl = self.sl(0)
        array *= (2/np.pi)
        array[sl] /= 2
        if self.quad == 'GL':
            sl[self.axis] = -1
            array[sl] /= 2
        assert array is self.xfftn_fwd.output_array
        return array

    def evaluate_expansion_all(self, fk, fj):
        fj = self.xfftn_bck()

        s0 = self.sl(slice(0, 1))
        if self.quad == "GC":
            fj *= 0.5
            fj += fk[s0]/2

        elif self.quad == "GL":
            fj *= 0.5
            fj += fk[s0]/2
            s0[self.axis] = slice(-1, None)
            s2 = self.sl(slice(0, None, 2))
            fj[s2] += fk[s0]/2
            s2[self.axis] = slice(1, None, 2)
            fj[s2] -= fk[s0]/2

        assert fk is self.xfftn_bck.input_array
        assert fj is self.xfftn_bck.output_array

        return fj

    def scalar_product(self, input_array=None, output_array=None, fast_transform=True):
        if input_array is not None:
            self.scalar_product.input_array[...] = input_array

        if fast_transform:
            assert self.N == self.scalar_product.input_array.shape[self.axis]
            out = self.xfftn_fwd()
            if self.quad == "GC":
                out *= (np.pi/(2*self.N))

            elif self.quad == "GL":
                out *= (np.pi/(2*(self.N-1)))
            assert out is self.scalar_product.output_array

        else:
            self.vandermonde_scalar_product(self.scalar_product.input_array,
                                            self.scalar_product.output_array)

        if output_array is not None:
            output_array[...] = self.scalar_product.output_array
            return output_array
        else:
            return self.scalar_product.output_array

    def eval(self, x, fk):
        return n_cheb.chebval(x, fk)


@inheritdocstrings
class ShenDirichletBasis(ChebyshevBase):
    """Shen basis for Dirichlet boundary conditions

    kwargs:
        N             int         Number of quadrature points
        quad        ('GL', 'GC')  Chebyshev-Gauss-Lobatto or Chebyshev-Gauss
        bc             (a, b)     Boundary conditions at x=(1,-1)

    """

    def __init__(self, N=0, quad="GC", bc=(0., 0.), plan=False):
        ChebyshevBase.__init__(self, N, quad)
        self.bc = bc
        self.CT = Basis(N, quad)
        if plan:
            self.plan(N, 0, np.float, {})

    def get_vandermonde_basis(self, V):
        P = np.zeros(V.shape)
        P[:, :-2] = V[:, :-2] - V[:, 2:]
        P[:, -2] = (V[:, 0] + V[:, 1])/2
        P[:, -1] = (V[:, 0] - V[:, 1])/2
        return P

    def scalar_product(self, input_array=None, output_array=None, fast_transform=True):
        if input_array is not None:
            self.scalar_product.input_array[...] = input_array

        if fast_transform:
            output = self.CT.scalar_product(fast_transform=fast_transform)
            s0 = self.sl(0)
            s1 = self.sl(1)
            c0 = 0.5*(output[s0] + output[s1])
            c1 = 0.5*(output[s0] - output[s1])
            s0[self.axis] = slice(0, -2)
            s1[self.axis] = slice(2, None)
            output[s0] -= output[s1]
            #output[:-2] -= output[2:]
            s0[self.axis] = -2
            output[s0] = c0
            s0[self.axis] = -1
            output[s0] = c1

        else:
            self.vandermonde_scalar_product(self.scalar_product.input_array,
                                            self.scalar_product.output_array)

        if output_array is not None:
            output_array[...] = self.scalar_product.output_array
            return output_array
        else:
            return self.scalar_product.output_array

    def evaluate_expansion_all(self, fk, fj):
        w_hat = work[(fk, 0)]
        s0 = self.sl(slice(0, -2))
        s1 = self.sl(slice(2, None))
        w_hat[s0] = fk[s0]
        w_hat[s1] -= fk[s0]
        s0[self.axis] = 0
        s1[self.axis] = 1
        w_hat[s0] += 0.5*(self.bc[0] + self.bc[1])
        w_hat[s1] += 0.5*(self.bc[0] - self.bc[1])
        fj = self.CT.backward(w_hat)
        assert fk is self.xfftn_bck.input_array
        assert fj is self.xfftn_bck.output_array
        return fj

    def forward(self, input_array=None, output_array=None, fast_transform=True):
        if input_array is not None:
            self.forward.input_array[...] = input_array

        output = self.scalar_product(fast_transform=fast_transform)
        assert output is self.forward.output_array
        s = self.sl(0)
        output[s] -= np.pi/2*(self.bc[0] + self.bc[1])
        s[self.axis] = 1
        output[s] -= np.pi/4*(self.bc[0] - self.bc[1])
        assert output is self.forward.output_array

        self.apply_inverse_mass(output)
        s[self.axis] = -2
        output[s] = self.bc[0]
        s[self.axis] = -1
        output[s] = self.bc[1]

        assert output is self.forward.output_array
        if output_array is not None:
            output_array[...] = self.forward.output_array
            return output_array
        else:
            return self.forward.output_array

    def slice(self):
        return slice(0, self.N-2)

    def eval(self, x, fk):
        w_hat = work[(fk, 0)]
        f = n_cheb.chebval(x, fk[:-2])
        w_hat[2:] = fk[:-2]
        f -= n_cheb.chebval(x, w_hat)
        return f + 0.5*(fk[-1]*(1+x)+fk[-2]*(1-x))

    def plan(self, shape, axis, dtype, options):
        self.CT.plan(shape, axis, dtype, options)
        self.axis = self.CT.axis
        self.xfftn_fwd = self.CT.xfftn_fwd
        self.xfftn_bck = self.CT.xfftn_bck
        self.forward = _func_wrap(self.forward, self.xfftn_fwd)
        self.backward = _func_wrap(self.backward, self.xfftn_bck)
        self.scalar_product = _func_wrap(self.scalar_product, self.xfftn_fwd)


@inheritdocstrings
class ShenNeumannBasis(ChebyshevBase):
    """Shen basis for homogeneous Neumann boundary conditions

    kwargs:
        N             int         Number of quadrature points
        quad        ('GL', 'GC')  Chebyshev-Gauss-Lobatto or Chebyshev-Gauss
        threads          1        Number of threads used by pyfftw
        planner_effort            Planner effort for FFTs.
        mean           float      Mean value

    """

    def __init__(self, N=0, quad="GC", mean=0, plan=False):
        ChebyshevBase.__init__(self, N, quad)
        self.mean = mean
        self.CT = Basis(N, quad)
        self._factor = np.zeros(0)
        if plan:
            self.plan(N, 0, np.float, {})

    def get_vandermonde_basis(self, V):
        assert self.N == V.shape[1]
        P = np.zeros(V.shape)
        k = np.arange(self.N).astype(np.float)
        P[:, :-2] = V[:, :-2] - (k[:-2]/(k[:-2]+2))**2*V[:, 2:]
        return P

    def set_factor_array(self, v):
        if not self._factor.shape == v.shape:
            k = self.wavenumbers(v.shape, self.axis)
            self._factor = (k/(k+2))**2

    def scalar_product(self, input_array=None, output_array=None, fast_transform=True):
        if input_array is not None:
            self.xfftn_fwd.input_array[...] = input_array

        if fast_transform:
            fk = self.CT.scalar_product(fast_transform=fast_transform)
            self.set_factor_array(fk)
            sm2 = self.sl(slice(0, -2))
            s2p = self.sl(slice(2, None))
            fk[sm2] -= self._factor * fk[s2p]

        else:
            self.vandermonde_scalar_product(self.xfftn_fwd.input_array,
                                            self.xfftn_fwd.output_array)

        s = self.sl(0)
        fk[s] = self.mean*np.pi
        s[self.axis] = slice(-2, None)
        fk[s] = 0

        if output_array is not None:
            output_array[...] = self.xfftn_fwd.output_array
            return output_array
        else:
            return self.xfftn_fwd.output_array

    def evaluate_expansion_all(self, fk, fj):
        w_hat = work[(fk, 0)]
        self.set_factor_array(fk)
        s0 = self.sl(slice(0, -2))
        s1 = self.sl(slice(2, None))
        w_hat[s0] = fk[s0]
        w_hat[s1] -= self._factor*fk[s0]
        fj = self.CT.backward(w_hat)
        return fj

    def slice(self):
        return slice(0, self.N-2)

    def eval(self, x, fk):
        w_hat = work[(fk, 0)]
        self.set_factor_array(fk)
        f = n_cheb.chebval(x, fk[:-2])
        w_hat[2:] = self._factor*fk[:-2]
        f -= n_cheb.chebval(x, w_hat)
        return f

    def plan(self, shape, axis, dtype, options):
        self.CT.plan(shape, axis, dtype, options)
        self.axis = self.CT.axis
        self.xfftn_fwd = self.CT.xfftn_fwd
        self.xfftn_bck = self.CT.xfftn_bck
        self.forward = _func_wrap(self.forward, self.xfftn_fwd)
        self.backward = _func_wrap(self.backward, self.xfftn_bck)
        self.scalar_product = _func_wrap(self.scalar_product, self.xfftn_fwd)


@inheritdocstrings
class ShenBiharmonicBasis(ChebyshevBase):
    """Shen biharmonic basis

    Homogeneous Dirichlet and Neumann boundary conditions.

    kwargs:
        N             int         Number of quadrature points
        quad        ('GL', 'GC')  Chebyshev-Gauss-Lobatto or Chebyshev-Gauss

    """

    def __init__(self, N=0, quad="GC", plan=False):
        ChebyshevBase.__init__(self, N, quad)
        self.CT = Basis(N, quad)
        self._factor1 = np.zeros(0)
        self._factor2 = np.zeros(0)
        if plan:
            self.plan(N, 0, np.float, {})

    def get_vandermonde_basis(self, V):
        P = np.zeros_like(V)
        k = np.arange(V.shape[1]).astype(np.float)[:-4]
        P[:, :-4] = V[:, :-4] - (2*(k+2)/(k+3))*V[:, 2:-2] + ((k+1)/(k+3))*V[:, 4:]
        return P

    def set_factor_arrays(self, v):
        s = [slice(None)]*v.ndim
        s[self.axis] = self.slice()
        if not self._factor1.shape == v[s].shape:
            k = self.wavenumbers(v.shape, axis=self.axis)
            self._factor1 = (-2*(k+2)/(k+3)).astype(float)
            self._factor2 = ((k+1)/(k+3)).astype(float)

    def scalar_product(self, input_array=None, output_array=None, fast_transform=True):
        if input_array is not None:
            self.scalar_product.input_array[...] = input_array

        if fast_transform:
            output = self.CT.scalar_product(fast_transform=fast_transform)
            Tk = work[(output, 0)]
            Tk[...] = output
            self.set_factor_arrays(Tk)

            s = self.sl(self.slice())
            s2 = self.sl(slice(2, -2))
            output[s] += self._factor1 * Tk[s2]
            s2[self.axis] = slice(4, None)
            output[s] += self._factor2 * Tk[s2]

        else:
            output = self.vandermonde_scalar_product(self.scalar_product.input_array,
                                                     self.scalar_product.output_array)

        output[self.sl(slice(-4, None))] = 0

        if output_array is not None:
            output_array[...] = output
            return output_array
        else:
            return output

    #@optimizer
    def set_w_hat(self, w_hat, fk, f1, f2):
        s = self.sl(self.slice())
        s2 = self.sl(slice(2, -2))
        s4 = self.sl(slice(4, None))
        w_hat[s] = fk[s]
        w_hat[s2] += f1*fk[s]
        w_hat[s4] += f2*fk[s]
        return w_hat

    def evaluate_expansion_all(self, fk, fj):
        w_hat = work[(fk, 0)]
        self.set_factor_arrays(fk)
        w_hat = self.set_w_hat(w_hat, fk, self._factor1, self._factor2)
        fj = self.CT.backward(w_hat)
        assert fk is self.backward.input_array
        assert fj is self.backward.output_array
        return fj

    def slice(self):
        return slice(0, self.N-4)

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

    def plan(self, shape, axis, dtype, options):
        self.CT.plan(shape, axis, dtype, options)
        self.axis = self.CT.axis
        self.xfftn_fwd = self.CT.xfftn_fwd
        self.xfftn_bck = self.CT.xfftn_bck
        self.forward = _func_wrap(self.forward, self.xfftn_fwd)
        self.backward = _func_wrap(self.backward, self.xfftn_bck)
        self.scalar_product = _func_wrap(self.scalar_product, self.xfftn_fwd)

