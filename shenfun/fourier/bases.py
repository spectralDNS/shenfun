"""
Module for defining bases in the Fourier family
"""
import numpy as np
from mpi4py_fft import fftw
from shenfun.spectralbase import SpectralBase, Transform, islicedict, slicedict
from shenfun.utilities import inheritdocstrings
from shenfun.optimization.cython import convolve

__all__ = ['FourierBase', 'R2CBasis', 'C2CBasis']

#pylint: disable=method-hidden, no-member, line-too-long, arguments-differ

@inheritdocstrings
class FourierBase(SpectralBase):
    r"""Fourier base class

    A basis function :math:`\phi_k` is given as

    .. math::

        \phi_k(x) = \exp(ikx)

    and an expansion is given as

    .. math::
       :label: u

        u(x) = \sum_k \hat{u}_k \exp(ikx)

    where

    .. math::

        k = -N/2, -N/2+1, ..., N/2-1

    However, since :math:`\exp(ikx) = \exp(i(k \pm N)x)` this expansion can
    also be written as an interpolator

    .. math::
       :label: u2

        u(x) = \sum_k \frac{\hat{u}_k}{c_k} \exp(ikx)

    where

    .. math::

        k = -N/2, -N/2+1, ..., N/2-1, N/2

    and :math:`c_{N/2} = c_{-N/2} = 2`, whereas :math:`c_k = 1` for
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

    Parameters
    ----------
        N : int
            Number of quadrature points. Should be even for efficiency, but
            this is not required.
        padding_factor : float, optional
                         Factor for padding backward transforms.
                         padding_factor=1.5 corresponds to a 3/2-rule for
                         dealiasing.
        domain : 2-tuple of floats, optional
                 The computational domain.
        dealias_direct : bool, optional
                         True for dealiasing using 2/3-rule. Must be used with
                         padding_factor == 1.
    """

    def __init__(self, N, padding_factor=1., domain=(0, 2*np.pi),
                 dealias_direct=False):
        self.dealias_direct = dealias_direct
        self._k = None
        self._planned_axes = None  # Collapsing of axes means that this base can be used to plan transforms over several collapsed axes. Store the axes planned for here.
        SpectralBase.__init__(self, N, '', padding_factor, domain)

    @staticmethod
    def family():
        return 'fourier'

    @staticmethod
    def boundary_condition():
        return 'Periodic'

    def points_and_weights(self, N=None, map_true_domain=False, **kw):
        if N is None:
            N = self.N
        points = np.arange(N, dtype=np.float)*2*np.pi/N
        if map_true_domain is True:
            points = self.map_true_domain(points)
        return points, np.array([2*np.pi/N])

    def evaluate_basis(self, x=None, i=0, output_array=None):
        if x is None:
            x = self.mesh(False, False)
        x = np.atleast_1d(x)
        if output_array is None:
            output_array = np.zeros(x.shape, dtype=np.complex)

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
            l = self.wavenumbers(bcast=False, scaled=True)
            V = V*((1j*l)**k)[np.newaxis, :]
        return V

    # Reimplemented for efficiency (smaller array in *= when truncated)
    #@profile
    def forward(self, input_array=None, output_array=None, fast_transform=True):
        if fast_transform is False:
            return SpectralBase.forward(self, input_array, output_array, False)

        if input_array is not None:
            self.forward.input_array[...] = input_array

        self.forward.xfftn()
        self._truncation_forward(self.forward.tmp_array,
                                 self.forward.output_array)
        M = self.get_normalization()
        self.forward._output_array *= M

        if output_array is not None:
            output_array[...] = self.forward.output_array
            return output_array
        return self.forward.output_array

    def apply_inverse_mass(self, array):
        """Apply inverse mass

        Note
        ----
        Mass matrix is identity, so do nothing

        Parameters
        ----------
            array : array (input/output)
                    Expansion coefficients.
        """
        return array

    def evaluate_expansion_all(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            SpectralBase.evaluate_expansion_all(self, input_array, output_array, False)
        else:
            self.backward.xfftn(normalise_idft=False)
        assert input_array is self.backward.xfftn.input_array

    def evaluate_scalar_product(self, input_array, output_array, fast_transform=True):
        if fast_transform is False:
            self.vandermonde_scalar_product(input_array, output_array)
            return
        output = self.scalar_product.xfftn()
        M = self.get_normalization()
        output *= M
        assert input_array is self.scalar_product.xfftn.input_array

    def vandermonde_scalar_product(self, input_array, output_array):
        SpectralBase.vandermonde_scalar_product(self, input_array, output_array)
        output_array *= 0.5/np.pi

    def reference_domain(self):
        return (0., 2*np.pi)

    @property
    def is_orthogonal(self):
        return True

    def get_orthogonal(self):
        return self

    def plan(self, shape, axis, dtype, options):
        if isinstance(axis, int):
            axis = [axis]
        s = tuple(np.take(shape, axis))

        if isinstance(self.forward, Transform):
            if self.forward.input_array.shape == shape and axis == self._planned_axes:
                # Already planned
                return

        plan_fwd = self._xfftn_fwd
        plan_bck = self._xfftn_bck

        if 'builders' in self._xfftn_fwd.__module__:

            opts = dict(
                avoid_copy=True,
                overwrite_input=True,
                auto_align_input=True,
                auto_contiguous=True,
                planner_effort='FFTW_MEASURE',
                threads=1,
            )
            opts.update(options)
            U = fftw.aligned(shape, dtype=dtype)
            xfftn_fwd = plan_fwd(U, s=s, axes=axis, **opts)
            V = xfftn_fwd.output_array
            xfftn_bck = plan_bck(V, s=s, axes=axis, **opts)
            V.fill(0)
            U.fill(0)

            xfftn_fwd.update_arrays(U, V)
            xfftn_bck.update_arrays(V, U)
            self._M = 1./np.prod(np.take(shape, axis))

        else:
            opts = dict(
                overwrite_input='FFTW_DESTROY_INPUT',
                planner_effort='FFTW_MEASURE',
                threads=1,
            )
            opts.update(options)
            flags = (fftw.flag_dict[opts['planner_effort']],
                     fftw.flag_dict[opts['overwrite_input']])
            threads = opts['threads']

            U = fftw.aligned(shape, dtype=dtype)
            xfftn_fwd = plan_fwd(U, s=s, axes=axis, threads=threads, flags=flags)
            V = xfftn_fwd.output_array

            if np.issubdtype(dtype, np.floating):
                flags = (fftw.flag_dict[opts['planner_effort']],)

            xfftn_bck = plan_bck(V, s=s, axes=axis, threads=threads, flags=flags, output_array=U)
            V.fill(0)
            U.fill(0)
            self._M = xfftn_fwd.get_normalization()
        self.axis = axis[-1]
        self._planned_axes = axis

        if self.padding_factor > 1.+1e-8:
            trunc_array = self._get_truncarray(shape, V.dtype)
            self.forward = Transform(self.forward, xfftn_fwd, U, V, trunc_array)
            self.backward = Transform(self.backward, xfftn_bck, trunc_array, V, U)
        else:
            self.forward = Transform(self.forward, xfftn_fwd, U, V, V)
            self.backward = Transform(self.backward, xfftn_bck, V, V, U)

        # scalar_product is not padded, just the forward/backward
        self.scalar_product = Transform(self.scalar_product, xfftn_fwd, U, V, V)
        self.si = islicedict(axis=self.axis, dimensions=self.dimensions)
        self.sl = slicedict(axis=self.axis, dimensions=self.dimensions)


class R2CBasis(FourierBase):
    """Fourier basis class for real to complex transforms
    """

    def __init__(self, N, padding_factor=1., domain=(0., 2.*np.pi),
                 dealias_direct=False):
        FourierBase.__init__(self, N, padding_factor, domain, dealias_direct)
        self.N = N
        #self._xfftn_fwd = pyfftw.builders.rfftn
        #self._xfftn_bck = pyfftw.builders.irfftn
        self._xfftn_fwd = fftw.rfftn
        self._xfftn_bck = fftw.irfftn
        self._sn = []
        self._sm = []
        self.plan((int(np.floor(padding_factor*N)),), (0,), np.float, {})

    def wavenumbers(self, bcast=True, scaled=False, eliminate_highest_freq=False):
        k = np.fft.rfftfreq(self.N, 1./self.N).astype(int)
        if self.N % 2 == 0 and eliminate_highest_freq:
            k[-1] = 0
        if scaled:
            k = k*self.domain_factor()
        if bcast is True:
            k = self.broadcast_to_ndims(k)
        return k

    def mask_nyquist(self, bcast=True):
        f = np.ones(self.N//2+1, dtype=int)
        if self.N % 2 == 0:
            f[-1] = 0
        if bcast is True:
            f = self.broadcast_to_ndims(f)
        return f

    def _get_truncarray(self, shape, dtype):
        shape = list(shape)
        shape[self.axis] = int(shape[self.axis] / self.padding_factor)
        shape[self.axis] = shape[self.axis]//2 + 1
        return fftw.aligned(shape, dtype=dtype)

    def slice(self):
        return slice(0, self.N//2+1)

    def shape(self, forward_output=True):
        if forward_output:
            return self.N//2+1
        return self.N

    def vandermonde_evaluate_expansion_all(self, input_array, output_array):
        assert abs(self.padding_factor-1) < 1e-8
        assert self.N == output_array.shape[self.axis]
        points = self.points_and_weights()[0]
        P = self.vandermonde(points)
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

    def vandermonde_evaluate_expansion(self, points, input_array, output_array):
        """Evaluate expansion at certain points, possibly different from
        the quadrature points

        This method assumes the array is locally available in full, i.e., the
        multidimensional arrays are aligned along the axis of this basis.

        Parameters
        ----------
            P : 2D array
                Vandermode matrix containing local points only
            input_array : array
                          Expansion coefficients
            output_array : array
                           Function values on points
            last_conj_index : int
                              The last index to sum over for conj part
                              (R2CBasis only)
            offset : int
                     Global offset (MPI)

        Note
        ----
        This method is complicated by the fact that the data may not be aligned
        along the axis of this basis.

        """
        assert abs(self.padding_factor-1) < 1e-8
        points = self.map_reference_domain(points)
        P = self.evaluate_basis_all(points)
        assert output_array.ndim == 1 # Multidimensional should use vandermonde_evaluate_local_expansion
        output_array[:] = np.dot(P, input_array).real
        if self.N % 2 == 0:
            output_array += np.conj(np.dot(P[:, 1:-1], input_array[1:-1])).real
        else:
            output_array += np.conj(np.dot(P[:, 1:], input_array[1:])).real

        return output_array

    def _truncation_forward(self, padded_array, trunc_array):
        if self.padding_factor > 1.0+1e-8:
            trunc_array.fill(0)
            N = trunc_array.shape[self.axis]
            s = self.sl[slice(0, N)]
            trunc_array[:] = padded_array[s]
            if self.N % 2 == 0:
                s1 = self.si[N-1]
                trunc_array[s1] = trunc_array[s1].real
                trunc_array[s1] *= 2

    def _padding_backward(self, trunc_array, padded_array):
        if self.padding_factor > 1.0+1e-8:
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
            convolve.convolve_real_1D(u, v, uv, k1)

            #u1 = np.hstack((u, np.conj(u[1:][::-1])))
            #if N % 2 == 0:
                #u1[N//2:N//2+2] *= 0.5
            #v1 = np.hstack((v, np.conj(v[1:][::-1])))
            #if N % 2 == 0:
                #v1[N//2:N//2+2] *= 0.5

            #for m in range(N):
                #vc = np.roll(v1, -(m+1))
                #s = u1*vc[::-1]
                #ki = k1 + np.roll(k1, -(m+1))[::-1]
                #z0 = np.argwhere(ki == m)
                #z1 = np.argwhere(ki == m-N)
                #uv[m] = np.sum(s[z0])
                #uv[m-N] = np.sum(s[z1])

            #for m in k1:
                #for n in k1:
                    #p = m + n
                    #if p >= 0:
                        #if N % 2 == 0:
                            #if abs(m) == N//2:
                                #um = u[abs(m)]*0.5
                            #elif m >= 0:
                                #um = u[m]
                            #else:
                                #um = np.conj(u[abs(m)])
                            #if abs(n) == N//2:
                                #vn = v[abs(n)]*0.5
                            #elif n >= 0:
                                #vn = v[n]
                            #else:
                                #vn = np.conj(v[abs(n)])
                        #else:
                            #if m >= 0:
                                #um = u[m]
                            #elif m < 0:
                                #um = np.conj(u[abs(m)])
                            #if n >= 0:
                                #vn = v[n]
                            #elif n < 0:
                                #vn = np.conj(v[abs(n)])
                        #uv[p] += um*vn
        return uv


class C2CBasis(FourierBase):
    """Fourier basis class for complex to complex transforms
    """

    def __init__(self, N, padding_factor=1., domain=(0., 2.*np.pi),
                 dealias_direct=False):
        FourierBase.__init__(self, N, padding_factor, domain, dealias_direct)
        self.N = N
        #self._xfftn_fwd = pyfftw.builders.fftn
        #self._xfftn_bck = pyfftw.builders.ifftn
        self._xfftn_fwd = fftw.fftn
        self._xfftn_bck = fftw.ifftn
        self.plan((int(np.floor(padding_factor*N)),), (0,), np.complex, {})
        self._slp = []

    def wavenumbers(self, bcast=True, scaled=False, eliminate_highest_freq=False):
        k = np.fft.fftfreq(self.N, 1./self.N).astype(int)
        if self.N % 2 == 0 and eliminate_highest_freq:
            k[self.N//2] = 0
        if scaled:
            k = k*self.domain_factor()
        if bcast is True:
            k = self.broadcast_to_ndims(k)
        return k

    def mask_nyquist(self, bcast=True):
        f = np.ones(self.N, dtype=int)
        if self.N % 2 == 0:
            f[self.N//2] = 0
        if bcast is True:
            f = self.broadcast_to_ndims(f)
        return f

    def slice(self):
        return slice(0, self.N)

    def _truncation_forward(self, padded_array, trunc_array):
        if self.padding_factor > 1.0+1e-8:
            trunc_array.fill(0)
            N = trunc_array.shape[self.axis]
            su = self.sl[slice(0, N//2+1)]
            trunc_array[su] = padded_array[su]
            su = self.sl[slice(-(N//2), None)]
            trunc_array[su] += padded_array[su]

    def _padding_backward(self, trunc_array, padded_array):
        if self.padding_factor > 1.0+1e-8:
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
            convolve.convolve_1D(u, v, uv, k)

            #if N % 2 == 0:
                #u = np.hstack((u[:N//2], u[N//2], u[N//2:]))
                #u[N//2:N//2+2] *= 0.5
                #v = np.hstack((v[:N//2], v[N//2], v[N//2:]))
                #v[N//2:N//2+2] *= 0.5

            #for m in range(Np):
                #vc = np.roll(v, -(m+1))
                #s = u*vc[::-1]
                #ki = k + np.roll(k, -(m+1))[::-1]
                #z0 = np.argwhere(ki == m)
                #z1 = np.argwhere(ki == m-Np)
                #uv[m] = np.sum(s[z0])
                #uv[m-Np] = np.sum(s[z1])

            #for m in k:
                #for n in k:
                    #p = m + n
                    #if N % 2 == 0:
                        #if abs(m) == N//2:
                            #um = u[m]*0.5
                        #else:
                            #um = u[m]
                        #if abs(n) == N//2:
                            #vn = v[n]*0.5
                        #else:
                            #vn = v[n]
                    #else:
                        #um = u[m]
                        #vn = v[n]
                    #uv[p] += um*vn

        return uv
