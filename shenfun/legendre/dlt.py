import numpy as np
from numpy.polynomial import chebyshev as n_cheb
from mpi4py_fft import fftw
from mpi4py_fft.fftw.utilities import FFTW_MEASURE, FFTW_PRESERVE_INPUT
from . import fastgl

__all__ = ['DLT']

class DLT:
    """Discrete Legendre transform

    A class for performing fast FFT-based discrete Legendre transforms, both
    forwards and backwards. Based on::

        Nicholas Hale and Alex Townsend "A fast FFT-based discrete Legendre
        transform", IMA Journal of Numerical Analysis (2015)
        (https://arxiv.org/abs/1505.00354)

    Parameters
    ----------
    input_array : real or complex array
    s : sequence of ints, optional
        Not used - included for compatibility with Numpy
    axes : integer or 1-tuple of int, optional
        Axis over which to compute the DLT. Named axes for compatibility.
    threads : int, optional
        Number of threads used in computing DLT.
    kind : str, optional
        Either one of

            - 'forward'
            - 'backward'
            - 'scalar product'
    flags : sequence of ints, optional
        Flags from

            - FFTW_MEASURE
            - FFTW_EXHAUSTIVE
            - FFTW_PATIENT
            - FFTW_DESTROY_INPUT
            - FFTW_PRESERVE_INPUT
            - FFTW_UNALIGNED
            - FFTW_CONSERVE_MEMORY
            - FFTW_ESTIMATE
    output_array : complex array, optional
        Array to be used as output array. Must be of correct shape, type,
        strides and alignment

    Note
    ----
    The Legendre series expansion of :math:`f(x)` is

    .. math::

        f(x) = \sum_{k=0}^{N-1} \hat{f}_k L_k(x)

    The series evaluated at quadrature points :math:`\{x_j\}_{j=0}^{N-1}` gives the
    vector :math:`\{f(x_j)\}_{j=0}^{N-1}`. We define a forward transform as
    computing :math:`\{\hat{f}_k\}_{k=0}^{N-1}` from :math:`\{f(x_j)\}_{j=0}^{N-1}`.
    The other way around, a backward transform, computes :math:`\{f(x_j)\}_{j=0}^{N-1}`
    given :math:`\{\hat{f}_k\}_{k=0}^{N-1}`. This is in agreement with shenfun's
    definition of forward/backward directions, but it disagrees with the definitions
    used by Hale and Townsend.

    Also note that this `fast` transform is actually slower than the default recursive
    version for approximately :math:`N<1000`.

    Example
    -------
    >>> import numpy as np
    >>> from shenfun import legendre, FunctionSpace, Function, Array
    >>> N = 4
    >>> L = FunctionSpace(N, 'L')
    >>> u = Function(L)
    >>> u[1] = 1
    >>> c = Array(L)
    >>> c = L.backward(u, c, kind='fast')
    >>> print(c)
    [-0.86113631 -0.33998104  0.33998104  0.86113631]
    >>> u[:] = 1
    >>> np.alltrue(np.abs(u - u.backward().forward()) < 1e-12)
    True

    """
    def __init__(self, input_array, s=None, axes=(-1,), threads=1, kind='backward',
                 flags=(FFTW_MEASURE, FFTW_PRESERVE_INPUT), output_array=None):
        if isinstance(axes, tuple):
            assert len(axes) == 1
            axis = axes[-1]
        elif isinstance(axes, int):
            axis = axes
        self.axis = axis
        assert kind in ('forward', 'scalar product', 'backward')
        self.kind = kind
        N = self.N = input_array.shape[axis]
        xl, self.wl = fastgl.leggauss(N)
        xc = n_cheb.chebgauss(N)[0]
        thetal = np.arccos(xl)[::-1]
        thetac = np.arccos(xc)
        s = [None]*input_array.ndim
        s[axis] = slice(None)
        self.dtheta = (thetal - thetac)[tuple(s)]
        self.n = np.arange(N)[tuple(s)]
        self.wl = self.wl[tuple(s)]
        s0 = [slice(None)]*input_array.ndim
        s0[self.axis] = slice(-1, None, -1) # reverse
        self.s0 = tuple(s0)
        if kind in ('forward', 'scalar product'):
            self.nsign = np.ones(N)
            self.nsign[1::2] = -1
            if kind == 'forward': # apply inverse mass matrix
                self.nsign = self.nsign[tuple(s)]*(self.n+0.5)
        self.M2 = None
        self.M2T = None
        U = input_array
        V = output_array if output_array is not None else U.copy()
        self.plan(U, V, kind, threads, flags)

    def plan(self, U, V, kind, threads, flags):
        if kind in ('forward', 'scalar product'):
            self.dct = DCT(U, axis=self.axis, type=2, threads=threads, flags=flags, output_array=V)
            self.dst = DST(U, axis=self.axis, type=2, threads=threads, flags=flags, output_array=V)
        else:
            self.dct = DCT(U, axis=self.axis, type=3, threads=threads, flags=flags, output_array=V)
            self.dst = DST(U, axis=self.axis, type=3, threads=threads, flags=flags, output_array=V)
        self._input_array = U
        self._output_array = V

    def assemble(self):
        # Create matrices for transforming between Legendre and Chebyshev coefficients
        # Note, this is a weakness, since the matrices are triangular, and
        # matrix vector products cost about 1/4 N^2.
        from shenfun import extract_diagonal_matrix, FunctionSpace, spectralbase, \
            SparseMatrix, inner, TestFunction, TrialFunction

        N = self.N
        T = FunctionSpace(N, 'C')
        L = FunctionSpace(N, 'L')
        ck = spectralbase.get_norm_sq(T, T, 'exact')
        MM = inner(TestFunction(T), TrialFunction(L))
        B = SparseMatrix({0: 1/ck}, (N, N))
        M2 = B.diags('csr')*MM.diags('csr')
        self.M2 = extract_diagonal_matrix(M2, lowerband=0)
        self.M2T = extract_diagonal_matrix(M2.T, upperband=0)

    def leg2cheb(self, f):
        f = self.M2.matvec(f.copy(), f, axis=self.axis, format='python')
        return f

    @property
    def input_array(self):
        return self._input_array

    @property
    def output_array(self):
        return self._output_array

    def __call__(self, input_array=None, output_array=None, **kw):
        if input_array is not None:
            self._input_array[:] = input_array
        if self.M2 is None:
            self.assemble()
        if self.kind in ('forward', 'scalar product'):
            self._input_array *= self.wl
        else:
            self._input_array = self.leg2cheb(self._input_array)
        fk = self.output_array.copy()
        fk[:] = self.dct(self._input_array)
        lf = 1
        c = self.input_array.copy()
        Dt = 1
        for l in range(1, 40):
            lf *= l
            if self.kind == 'backward':
                c *= self.n
                Dt *= self.dtheta
            else:
                c *= self.dtheta
                Dt *= self.n
            sign = (-1)**((l+1)//2)
            if l % 2 == 1:
                df = sign/lf*Dt*self.dst(c)
            else:
                df = sign/lf*Dt*self.dct(c)
            fk += df
            error = np.linalg.norm(df, ord=np.inf)
            if error < 1e-14:
                break
        if self.kind in ('forward', 'scalar product'):
            fk = self.M2T.matvec(fk.copy(), fk, axis=self.axis, format='python')
            fk *= self.nsign
        else:
            fk[:] = fk[self.s0]
        if output_array is not None:
            output_array[:] = fk
            return output_array
        self._output_array[:] = fk
        return self._output_array

class DCT:
    def __init__(self, input_array, type=3, s=None, axis=-1, threads=1,
                 flags=(FFTW_MEASURE,), output_array=None):
        self.axis = axis
        self.type = type
        self.dct = fftw.dctn(input_array, axes=(axis,), type=type, threads=threads,
                             flags=flags, output_array=output_array)
        s = [slice(None)]*input_array.ndim
        s[self.axis] = 0
        self.s = tuple(s)

    @property
    def input_array(self):
        return self.dct.input_array

    @property
    def output_array(self):
        return self.dct.output_array

    def __call__(self, input_array=None, output_array=None):
        if input_array is not None:
            self.input_array[:] = input_array
        out = self.dct()
        if self.type == 3:
            out += self.dct.input_array[self.s]
        out /= 2
        if output_array is not None:
            output_array[:] = out
            return output_array
        return out

class DST:
    def __init__(self, input_array, type=3, s=None, axis=-1, threads=1,
                 flags=None, output_array=None):
        self.axis = axis
        self.type = type
        self.dst = fftw.dstn(input_array, axes=(axis,), type=type, threads=threads,
                             flags=flags, output_array=output_array)
        self.sm1 = [slice(None)]*input_array.ndim
        self.sm1[axis] = slice(0, -1)
        self.sm1 = tuple(self.sm1)
        self.sp1 = [slice(None)]*input_array.ndim
        self.sp1[axis] = slice(1, None)
        self.sp1 = tuple(self.sp1)
        self.y = np.zeros_like(input_array)
        s = [slice(None)]*input_array.ndim
        s[self.axis] = 0
        self.s0 = tuple(s)

    @property
    def input_array(self):
        return self.dst.input_array

    @property
    def output_array(self):
        return self.dst.output_array

    def __call__(self, input_array=None, output_array=None):
        if input_array is not None:
            self.input_array[:] = input_array
        x = self.input_array
        if self.type == 3:
            self.y[self.sm1] = x[self.sp1]
            out = self.dst(self.y)
        elif self.type == 2:
            out = self.dst()
            out[self.sp1] = out[self.sm1]
            out[self.s0] = 0
        out /= 2
        if output_array is not None:
            output_array[:] = out
            return output_array
        return out
