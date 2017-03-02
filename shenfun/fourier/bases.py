import numpy as np
from mpiFFT4py import rfft, fft, irfft, ifft
from shenfun.spectralbase import SpectralBase, work
from shenfun.utilities import inheritdocstrings

__all__ = ['FourierBase', 'R2CBasis', 'C2CBasis']

@inheritdocstrings
class FourierBase(SpectralBase):
    """Fourier base class
    """

    def __init__(self, N, threads=1, planner_effort="FFTW_MEASURE"):
        SpectralBase.__init__(self, N, '')
        self.N = N
        self.threads = threads
        self.planner_effort = planner_effort

    def points_and_weights(self):
        """Return points and weights of quadrature"""
        points = np.arange(self.N, dtype=np.float)*2*np.pi/self.N
        return points, 2*np.pi/self.N

    def vandermonde(self, x):
        """Return Vandermonde matrix

        args:
            x               points for evaluation

        """
        k = self.wavenumbers(self.N)
        return np.exp(1j*x[:, np.newaxis]*k[np.newaxis, :])

    def get_vandermonde_basis_derivative(self, V, d=0):
        """Return k'th derivative of basis as a Vandermonde matrix

        args:
            V               Chebyshev Vandermonde matrix

        kwargs:
            k    integer    k'th derivative

        """
        if d > 0:
            k = self.wavenumbers(self.N)
            V = V*((1j*k)**d)[np.newaxis, :]
        return V

    def get_mass_matrix(self):
        from .matrices import mat
        return mat[(self.__class__, 0), (self.__class__, 0)]

    def apply_inverse_mass(self, fk, axis=0):
        """Apply inverse mass, which is identity for Fourier basis

        args:
            fk   (input/output)    Expansion coefficients

        kwargs:
            axis        int        The axis to apply inverse mass along

        """
        return fk/(2*np.pi)

class R2CBasis(FourierBase):
    """Fourier basis class for real to complex transforms
    """

    def __init__(self, N, threads=1, planner_effort="FFTW_MEASURE"):
        FourierBase.__init__(self, N, '')
        self.N = N
        self.threads = threads
        self.planner_effort = planner_effort

    def wavenumbers(self, N, axis=0):
        """Return the wavenumbermesh

        All dimensions, except axis, are obtained through broadcasting.

        """
        N = list(N) if np.ndim(N) else [N]
        assert self.N == N[axis]
        s = [np.newaxis]*len(N)
        s[axis] = self.slice()
        k = np.fft.rfftfreq(N[axis], 1./N[axis])
        return k[s]

    def evaluate_expansion_all(self, fk, fj, axis=0):
        fj = irfft(fk, fj, axis=axis, threads=self.threads,
                   planner_effort=self.planner_effort)
        fj *= self.N
        return fj

    def scalar_product(self, fj, fk, fast_transform=True, axis=0):
        if fast_transform:
            fk = rfft(fj, fk, axis=axis, threads=self.threads,
                      planner_effort=self.planner_effort)
            fk *= (2*np.pi/self.N)

        else:
            fk = self.vandermonde_scalar_product(fj, fk, axis=axis)

        return fk

    def eval(self, x, fk):
        V = self.vandermonde(x)
        return np.dot(V, fk) + np.conj(np.dot(V[:, 1:], fk[1:]))

    def slice(self):
        return slice(0, self.N//2+1)

    def get_shape(self):
        return self.N//2+1

    def vandermonde_evaluate_expansion_all(self, fk, fj, axis=0):
        """Naive implementation of evaluate_expansion_all

        args:
            fk   (input)    Expansion coefficients
            fj   (output)   Function values on quadrature mesh

        """
        assert self.N == fj.shape[axis]
        points = self.points_and_weights()[0]
        V = self.vandermonde(points)
        P = self.get_vandermonde_basis(V)
        if fj.ndim == 1:
            fj[:] = np.dot(P, fk).real
            fj += np.dot(P[:, 1:], np.conj(fk[1:])).real
        else:
            fc = np.moveaxis(fk, axis, -2)
            fj[:] = np.dot(P, fc).real
            fj += np.dot(P[:, 1:], np.conj(fc[1:])).real
            fj = np.moveaxis(fj, 0, axis)
        return fj


class C2CBasis(FourierBase):
    """Fourier basis class for complex to complex transforms
    """

    def __init__(self, N, threads=1, planner_effort="FFTW_MEASURE"):
        FourierBase.__init__(self, N, '')
        self.N = N
        self.threads = threads
        self.planner_effort = planner_effort

    def wavenumbers(self, N, axis=0):
        """Return the wavenumbermesh

        All dimensions, except axis, are obtained through broadcasting.

        """
        N = list(N) if np.ndim(N) else [N]
        assert self.N == N[axis]
        s = [np.newaxis]*len(N)
        s[axis] = self.slice()
        k = np.fft.fftfreq(N[axis], 1./N[axis])
        return k[s]

    def evaluate_expansion_all(self, fk, fj, axis=0):
        fj = ifft(fk, fj, axis=axis, threads=self.threads,
                  planner_effort=self.planner_effort)
        fj *= self.N
        return fj

    def scalar_product(self, fj, fk, fast_transform=True, axis=0):
        if fast_transform:
            fk = fft(fj, fk, axis=axis, threads=self.threads,
                     planner_effort=self.planner_effort)
            fk *= (2*np.pi/self.N)
        else:
            fk = self.vandermonde_scalar_product(fj, fk, axis=axis)
        return fk

    def eval(self, x, fk):
        V = self.vandermonde(x)
        return np.dot(V, fk)

    def slice(self):
        return slice(0, self.N)

    def get_shape(self):
        return self.N

    def vandermonde_evaluate_expansion_all(self, fk, fj, axis=0):
        """Naive implementation of evaluate_expansion_all

        args:
            fk   (input)    Expansion coefficients
            fj   (output)   Function values on quadrature mesh

        """
        assert self.N == fj.shape[axis]
        points = self.points_and_weights()[0]
        V = self.vandermonde(points)
        P = self.get_vandermonde_basis(V)
        if fj.ndim == 1:
            fj = np.dot(P, fk, out=fj)
        else:
            fc = np.moveaxis(fk, axis, -2)
            fj = np.dot(P, fc, out=fj)
            fj = np.moveaxis(fj, 0, axis)
        return fj
