r"""
Module for function spaces using Fourier exponentials.

A basis function :math:`\phi_k` is given as

.. math::
    \phi_k(x) = \exp(ikx)

and an expansion is given as

.. math::
    :label: u

    u(x) = \sum_{k=-N/2}^{N/2-1} \hat{u}_k \exp(ikx)

However, since :math:`\exp(ikx) = \exp(i(k \pm N)x)` this expansion can
also be written as an interpolator

.. math::
    :label: u2

    u(x) = \sum_{k=-N/2}^{N/2} \frac{\hat{u}_k}{c_k} \exp(ikx)

where :math:`c_{N/2} = c_{-N/2} = 2`, whereas :math:`c_k = 1` for
:math:`k=-N/2+1, ..., N/2-1`. Furthermore,
:math:`\hat{u}_{N/2} = \hat{u}_{-N/2}`.

The interpolator form is used for computing odd derivatives. Otherwise,
it makes no difference and therefore :eq:`u` is used in transforms, since
this is the form expected by fftw.
The inner product is defined as

.. math::
    (u, v) = \frac{1}{L} \int_{0}^{L} u \overline{v} dx

where :math:`\overline{v}` is the complex conjugate of :math:`v`, and
:math:`L` is the length of the (periodic) domain.
"""
import sympy as sp
import numpy as np
from mpi4py_fft import fftw
from shenfun.spectralbase import SpectralBase, Transform, islicedict, slicedict
from shenfun.optimization import cython
from shenfun.config import config

bases = ['R2C', 'C2C']
bcbases = []
testbases = []
__all__ = bases

#pylint: disable=method-hidden, no-member, line-too-long, arguments-differ


class FourierBase(SpectralBase):
    r"""Abstract base class for Fourier exponentials
    """
    def __init__(self, N, padding_factor=1, domain=(0, 2*sp.pi), dtype=float,
                 dealias_direct=False, coordinates=None):
        self._k = None
        self._planned_axes = None  # Collapsing of axes means that this base can be used to plan transforms over several collapsed axes. Store the axes planned for here.
        SpectralBase.__init__(self, N, dtype=dtype, padding_factor=padding_factor,
                              dealias_direct=dealias_direct, domain=domain,
                              coordinates=coordinates)

    @staticmethod
    def family():
        return 'fourier'

    @staticmethod
    def boundary_condition():
        return 'Periodic'

    def points_and_weights(self, N=None, map_true_domain=False, weighted=True, **kw):
        if N is None:
            N = self.shape(False)
        points = np.arange(N, dtype=float)*2*np.pi/N
        if map_true_domain is True:
            points = self.map_true_domain(points)
        if weighted:
            # weight is 1/self.domain_length() since this leads to the Kronecker delta function for mass matrix
            return points, np.array([float(self.domain_factor())/N])
        return points, np.array([2*np.pi/N])

    def orthogonal_basis_function(self, i=0, x=sp.symbols('x', real=True)):
        k = self.wavenumbers(bcast=False, scaled=False, eliminate_highest_freq=False)
        return sp.exp(sp.I*k[i]*x)

    def L2_norm_sq(self, i):
        return 1

    def l2_norm_sq(self, i=None):
        if i is None:
            return np.ones(self.N)
        return 1

    def weight(self, x=sp.symbols('x', real=True)):
        return 1/self.domain_length()

    def evaluate_basis(self, x=None, i=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=complex)

        if self._k is None:
            self._k = self.wavenumbers(bcast=False)
        k = self._k[i]
        output_array[:] = np.exp(1j*x*k)
        return output_array

    def evaluate_basis_derivative(self, x=None, i=0, k=0, output_array=None):
        output_array = self.evaluate_basis(x, i, output_array)
        l = self._k[i]
        output_array *= ((1j*l)**k)
        return output_array

    def vandermonde(self, x):
        k = self.wavenumbers(bcast=False)
        x = np.atleast_1d(x)
        return np.exp(1j*x[:, np.newaxis]*k[np.newaxis, :])

    def evaluate_basis_derivative_all(self, x=None, k=0):
        V = self.evaluate_basis_all(x=x)
        if k > 0:
            l = self.wavenumbers(bcast=False, scaled=False, eliminate_highest_freq=False)
            V = V*((1j*l)**k)[np.newaxis, :]
        return V

    # Reimplemented for efficiency (smaller array in *= when truncated)
    def forward(self, input_array=None, output_array=None, kind='fast'):
        if kind != 'fast':
            return SpectralBase.forward(self, input_array, output_array, kind=kind)

        if input_array is not None:
            self.forward.input_array[...] = input_array

        self.forward.xfftn()
        self._truncation_forward(self.forward.tmp_array,
                                 self.forward.output_array)
        M = self.get_normalization()
        self.forward._output_array *= M

        self.apply_inverse_mass(self.forward.output_array)

        if output_array is not None:
            output_array[...] = self.forward.output_array
            return output_array
        return self.forward.output_array

    def apply_inverse_mass(self, array):
        coors = self.tensorproductspace.coors if self.tensorproductspace else self.coors
        if not coors.is_cartesian: # mass matrix may not be diagonal, or there is scaling
            return SpectralBase.apply_inverse_mass(self, array)
        return array

    def _evaluate_scalar_product(self, kind='fast'):
        if kind != 'fast':
            SpectralBase._evaluate_scalar_product(self, kind=kind)
            return
        output = self.scalar_product.xfftn()
        output *= self.get_normalization()

    def reference_domain(self):
        return (0, 2*sp.pi)

    @property
    def is_orthogonal(self):
        return True

    def get_orthogonal(self, **kwargs):
        return self.get_unplanned(**kwargs)

    def get_homogeneous(self, **kwargs):
        return self.get_unplanned(**kwargs)

    def get_unplanned(self, **kwargs):
        d = dict(domain=self.domain,
                 padding_factor=self.padding_factor,
                 dealias_direct=self.dealias_direct,
                 coordinates=self.coors.coordinates)
        d.update(kwargs)
        return self.__class__(self.N, **d)

    def get_dealiased(self, padding_factor=1.5, dealias_direct=False, **kwargs):
        d = dict(domain=self.domain,
                 padding_factor=padding_factor,
                 dealias_direct=dealias_direct,
                 coordinates=self.coors.coordinates)
        d.update(kwargs)
        return self.__class__(self.N, **d)

    def get_refined(self, N, **kwargs):
        d = dict(domain=self.domain,
                 padding_factor=self.padding_factor,
                 dealias_direct=self.dealias_direct,
                 coordinates=self.coors.coordinates)
        d.update(kwargs)
        return self.__class__(N, **d)

    def mask_nyquist(self, u_hat, mask=None):
        """Return array `u_hat` with zero Nyquist coefficients

        Parameters
        ----------
        u_hat : array
            Array to be masked
        mask : array or None, optional
            mask array, if not provided then get the mask by calling
            :func:`get_mask_nyquist`
        """
        if mask is None:
            mask = self.get_mask_nyquist(bcast=False)
        if mask is not None:
            u_hat *= mask
        return u_hat

    def plan(self, shape, axis, dtype, options):
        if shape in (0, (0,)):
            return

        if isinstance(axis, int):
            axis = [axis]
        s = list(np.take(shape, axis))

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and axis == self._planned_axes:
                # Already planned
                return

        plan_fwd = self._xfftn_fwd
        plan_bck = self._xfftn_bck

        self.axis = axis[-1]
        self._planned_axes = axis

        #opts = dict(
        #    overwrite_input='FFTW_DESTROY_INPUT',
        #    planner_effort=self.opts['planner_effort'],
        #    threads=self.opts['threads'],
        #)
        opts = plan_fwd.opts
        opts['overwrite_input'] = 'FFTW_DESTROY_INPUT'
        opts.update(options)
        threads = opts['threads']
        flags = (fftw.flag_dict[opts['planner_effort']],
                 fftw.flag_dict[opts['overwrite_input']])

        U = fftw.aligned(shape, dtype=dtype)
        xfftn_fwd = plan_fwd(U, s=s, axes=axis, threads=threads, flags=flags)
        V = xfftn_fwd.output_array

        opts = plan_bck.opts
        opts['overwrite_input'] = 'FFTW_DESTROY_INPUT'
        opts.update(options)
        threads = opts['threads']
        flags = (fftw.flag_dict[opts['planner_effort']],
                 fftw.flag_dict[opts['overwrite_input']])
        if np.issubdtype(dtype, np.floating):
            flags = (fftw.flag_dict[opts['planner_effort']],)

        xfftn_bck = plan_bck(V, s=s, axes=axis, threads=threads, flags=flags, output_array=U)
        V.fill(0)
        U.fill(0)
        self._M = xfftn_fwd.get_normalization()

        if self.padding_factor > 1.+1e-8:
            trunc_array = self._get_truncarray(shape, V.dtype)
            self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, trunc_array)
            self.forward = Transform(self.forward, xfftn_fwd, U, V, trunc_array)
            self.backward = Transform(self.backward, xfftn_bck, trunc_array, V, U)

        else:
            self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)
            self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
            self.backward = Transform(self.backward, xfftn_bck, V, V, U)

        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)


class R2C(FourierBase):
    r"""Fourier function space for real to complex transforms

    A basis function :math:`\phi_k` is given as

    .. math::
        \phi_k(x) = \exp(ikx), \quad k =-N/2, -N/2+1, \ldots, N/2-1

    An expansion is given as

    .. math::
        u(x) = \sum_{k=-N/2}^{N/2-1} \hat{u}_k \exp(ikx).

    Howewer, due to Hermitian symmetry :math:`\hat{u}_k = \overline{\hat{u}_{-k}}`,
    which is taken advantage of in this function space. Specifically, Fourier
    transforms make use of real-to-complex and complex-to-real algorithms, see
    `FFTW <https://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html>`_
    and `rfftn <https://mpi4py-fft.readthedocs.io/en/latest/mpi4py_fft.fftw.html#mpi4py_fft.fftw.xfftn.rfftn>`_
    of `mpi4py-fft <https://github.com/mpi4py/mpi4py-fft>`_.

    Parameters
    ----------
    N : int
        Number of quadrature points. Should be even for efficiency, but
        this is not required.
    padding_factor : float, optional
        Factor for padding backward transforms. padding_factor=1.5
        corresponds to a 3/2-rule for dealiasing.
    domain : 2-tuple of numbers, optional
        The computational domain.
    dealias_direct : bool, optional
        True for dealiasing using 2/3-rule. Must be used with
        padding_factor = 1.
    coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
        Map for curvilinear coordinatesystem, and parameters to :class:`~shenfun.coordinates.Coordinates`

    """

    def __init__(self, N, padding_factor=1., domain=(0, 2*sp.pi),
                 dealias_direct=False, coordinates=None, **kw):
        FourierBase.__init__(self, N, padding_factor=padding_factor, dtype=float,
                             domain=domain, dealias_direct=dealias_direct,
                             coordinates=coordinates)
        self.N = N
        self._xfftn_fwd = fftw.rfftn
        self._xfftn_bck = fftw.irfftn
        self._xfftn_fwd.opts = config['fftw']['rfft']
        self._xfftn_bck.opts = config['fftw']['irfft']
        self._sn = []
        self._sm = []
        self.plan((int(padding_factor*N),), (0,), float, {})

    def wavenumbers(self, bcast=True, scaled=False, eliminate_highest_freq=False):
        k = np.fft.rfftfreq(self.N, 1./self.N).astype(int)
        if self.N % 2 == 0 and eliminate_highest_freq:
            k[-1] = 0
        if scaled:
            k = k*float(self.domain_factor())
        if bcast is True:
            k = self.broadcast_to_ndims(k)
        return k

    def get_mask_nyquist(self, bcast=True):
        """Return None or an array with zeros for Nyquist coefficients and one otherwise

        Parameters
        ----------
        bcast : boolean, optional
            If True then broadcast returned mask array to dimensions of the
            :class:`TensorProductSpace` this base belongs to.
        """
        if self.N % 2 == 0:
            f = np.ones(self.N//2+1, dtype=int)
            f[-1] = 0
        else:
            return None
        if bcast is True:
            f = self.broadcast_to_ndims(f)
        return f

    def _get_truncarray(self, shape, dtype):
        shape = list(shape)
        shape[self.axis] = int(shape[self.axis] / self.padding_factor)
        shape[self.axis] = shape[self.axis]//2 + 1
        return fftw.aligned(shape, dtype=dtype)

    @staticmethod
    def short_name():
        return 'R2C'

    def slice(self):
        return slice(0, self.N//2+1)

    def shape(self, forward_output=True):
        if forward_output:
            return self.N//2+1
        return int(np.floor(self.padding_factor*self.N))

    def _evaluate_expansion_all(self, input_array, output_array, x=None, kind='fast'):
        if kind == 'vandermonde':
            assert abs(self.padding_factor-1) < 1e-8
            P = self.evaluate_basis_all(x=x)
            if output_array.ndim == 1:
                output_array[:] = np.dot(P, input_array).real
                if self.N % 2 == 0:
                    output_array += np.conj(np.dot(P[:, 1:-1], input_array[1:-1])).real
                else:
                    output_array += np.conj(np.dot(P[:, 1:], input_array[1:])).real

            else:
                fc = np.moveaxis(input_array, self.axis, -2)
                array = np.dot(P, fc).real
                s = [slice(None)]*fc.ndim
                if self.N % 2 == 0:
                    s[-2] = slice(1, -1)
                    array += np.conj(np.dot(P[:, 1:-1], fc[tuple(s)])).real
                else:
                    s[-2] = slice(1, None)
                    array += np.conj(np.dot(P[:, 1:], fc[tuple(s)])).real

                output_array[:] = np.moveaxis(array, 0, self.axis)
            return
        assert kind == 'fast'
        assert input_array is self.backward.tmp_array
        assert output_array is self.backward.output_array
        self.backward.xfftn(normalise_idft=False)

    def _truncation_forward(self, padded_array, trunc_array):
        if not id(trunc_array) == id(padded_array):
            trunc_array.fill(0)
            N = trunc_array.shape[self.axis]
            s = self.sl[slice(0, N)]
            trunc_array[:] = padded_array[s]
            if self.N % 2 == 0:
                s1 = self.si[N-1]
                trunc_array[s1] = trunc_array[s1].real
                trunc_array[s1] *= 2

    def _padding_backward(self, trunc_array, padded_array):
        if not id(trunc_array) == id(padded_array):
            padded_array.fill(0)
            N = trunc_array.shape[self.axis]
            if len(self._sn) != self.dimensions:
                self._sn = self.sl[slice(0, N)]
                self._sm = self.si[N-1]
            padded_array[self._sn] = trunc_array[self._sn]
            if self.N % 2 == 0:  # Symmetric Fourier interpolator
                padded_array[self._sm] = padded_array[self._sm].real
                padded_array[self._sm] *= 0.5

        elif self.dealias_direct:
            N = self.N
            su = self.sl[slice(N//3, None)]
            padded_array[su] = 0

    def convolve(self, u, v, uv=None, fast=True):
        """Convolution of u and v.

        Parameters
        ----------
        u : array
        v : array
        uv : array, optional
        fast : bool, optional
               Whether to use fast transforms in computing convolution

        Note
        ----
        Note that this method is only valid for 1D data, and that
        for multidimensional arrays one should use corresponding method
        in the :class:`.TensorProductSpace` class.

        """
        N = self.N

        if fast is True:
            if uv is None:
                uv = self.forward.output_array.copy()

            assert self.padding_factor > 1.0, "padding factor must be > 3/2+1/N to perform convolution without aliasing"
            u2 = self.backward.output_array.copy()
            u3 = self.backward.output_array.copy()
            u2 = self.backward(u, u2)
            u3 = self.backward(v, u3)
            uv = self.forward(u2*u3, uv)

        else:
            if uv is None:
                uv = np.zeros(N+1, dtype=u.dtype)
            Np = N if not N % 2 == 0 else N+1
            k1 = np.fft.fftfreq(Np, 1./Np).astype(int)
            cython.convolve.convolve_real_1D(u, v, uv, k1)

        return uv


class C2C(FourierBase):
    r"""Fourier function space for complex to complex transforms

    A basis function :math:`\phi_k` is given as

    .. math::
        \phi_k(x) = \exp(ikx), \quad k =-N/2, -N/2+1, \ldots, N/2-1

    An expansion is given as

    .. math::
        u(x) = \sum_{k=-N/2}^{N/2-1} \hat{u}_k \exp(ikx).

    Parameters
    ----------
    N : int
        Number of quadrature points. Should be even for efficiency, but
        this is not required.
    padding_factor : float, optional
        Factor for padding backward transforms. padding_factor=1.5
        corresponds to a 3/2-rule for dealiasing.
    domain : 2-tuple of numbers, optional
        The computational domain.
    dealias_direct : bool, optional
        True for dealiasing using 2/3-rule. Must be used with
        padding_factor = 1.
    coordinates: 2- or 3-tuple (coordinate, position vector (, sympy assumptions)), optional
        Map for curvilinear coordinatesystem, and parameters to :class:`~shenfun.coordinates.Coordinates`

    """

    def __init__(self, N, padding_factor=1, domain=(0, 2*sp.pi),
                 dealias_direct=False, coordinates=None, **kw):
        FourierBase.__init__(self, N, padding_factor=padding_factor, dtype=complex,
                             domain=domain, dealias_direct=dealias_direct,
                             coordinates=coordinates)
        self.N = N
        self._xfftn_fwd = fftw.fftn
        self._xfftn_bck = fftw.ifftn
        self._xfftn_fwd.opts = config['fftw']['fft']
        self._xfftn_bck.opts = config['fftw']['ifft']
        self.plan((int(padding_factor*N),), (0,), complex, {})
        self._slp = []

    @staticmethod
    def short_name():
        return 'C2C'

    def _evaluate_expansion_all(self, input_array, output_array, x=None, kind='fast'):
        if kind == 'vandermonde':
            SpectralBase._evaluate_expansion_all(self, input_array, output_array, x, kind=kind)
            return
        assert kind == 'fast'
        assert input_array is self.backward.tmp_array
        assert output_array is self.backward.output_array
        self.backward.xfftn(normalise_idft=False)

    def wavenumbers(self, bcast=True, scaled=False, eliminate_highest_freq=False):
        k = np.fft.fftfreq(self.N, 1./self.N).astype(int)
        if self.N % 2 == 0 and eliminate_highest_freq:
            k[self.N//2] = 0
        if scaled:
            k = k*float(self.domain_factor())
        if bcast is True:
            k = self.broadcast_to_ndims(k)
        return k

    def get_mask_nyquist(self, bcast=True):
        if self.N % 2 == 0:
            f = np.ones(self.N, dtype=int)
            f[self.N//2] = 0
        else:
            return None
        if bcast is True:
            f = self.broadcast_to_ndims(f)
        return f

    def slice(self):
        return slice(0, self.N)

    def shape(self, forward_output=True):
        if forward_output:
            return self.N
        return int(np.floor(self.padding_factor*self.N))

    def count_trailing_zeros(self, u, reltol=1e-12, abstol=1e-15):
        assert u.function_space() == self
        assert u.ndim == 1
        a = abs(u[self.slice()])
        ua = (a < reltol*a.max()) | (a < abstol)
        return np.argmin(ua[self.N//2:]) + np.argmin(ua[:self.N//2][::-1])

    def _truncation_forward(self, padded_array, trunc_array):
        if not id(trunc_array) == id(padded_array):
            trunc_array.fill(0)
            N = trunc_array.shape[self.axis]
            su = self.sl[slice(0, N//2+1)]
            trunc_array[su] = padded_array[su]
            su = self.sl[slice(-(N//2), None)]
            trunc_array[su] += padded_array[su]

    def _padding_backward(self, trunc_array, padded_array):
        # pylint: disable=attribute-defined-outside-init
        if not id(trunc_array) == id(padded_array):
            padded_array.fill(0)
            N = trunc_array.shape[self.axis]
            if len(self._slp) != self.dimensions: # Store for microoptimization
                self._slp = self.sl[slice(0, N//2+1)]
                self._slm = self.sl[slice(-(N//2), None)]
                self._slp0 = self.si[N//2]
                self._slm0 = self.si[-(N//2)]
            padded_array[self._slp] = trunc_array[self._slp]
            padded_array[self._slm] = trunc_array[self._slm]
            if self.N % 2 == 0:  # Use symmetric Fourier interpolator
                padded_array[self._slp0] *= 0.5
                padded_array[self._slm0] *= 0.5

        elif self.dealias_direct:
            N = trunc_array.shape[self.axis]
            su = self.sl[slice(N//3, -(N//3)+1)]
            padded_array[su] = 0

    def convolve(self, u, v, uv=None, fast=True):
        """Convolution of u and v.

        Parameters
        ----------
            u : array
            v : array
            uv : array, optional
            fast : bool, optional
                   Whether to use fast transforms in computing convolution

        Note
        ----
        Note that this method is only valid for 1D data, and that
        for multidimensional arrays one should use corresponding method
        in the TensorProductSpace class.

        """
        assert len(u.shape) == 1
        N = self.N

        if fast:
            if uv is None:
                uv = self.forward.output_array.copy()

            assert self.padding_factor > 1.0, "padding factor must be > 3/2+1/N to perform convolution without aliasing"
            u2 = self.backward.output_array.copy()
            u3 = self.backward.output_array.copy()
            u2 = self.backward(u, u2)
            u3 = self.backward(v, u3)
            uv = self.forward(u2*u3, uv)

        else:

            if uv is None:
                uv = np.zeros(2*N, dtype=u.dtype)

            Np = N if not N % 2 == 0 else N+1
            k = np.fft.fftfreq(Np, 1./Np).astype(int)
            cython.convolve.convolve_1D(u, v, uv, k)

        return uv
