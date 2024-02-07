import importlib
from copy import copy
from scipy.special import gammaln
import numpy as np
from numpy.polynomial import chebyshev as n_cheb
from mpi4py import MPI
from mpi4py_fft import fftw
from mpi4py_fft.fftw.utilities import FFTW_MEASURE, FFTW_PRESERVE_INPUT
from shenfun.optimization import runtimeoptimizer
from shenfun.optimization import cython
from shenfun.spectralbase import islicedict, slicedict

__all__ = ['DLT', 'leg2cheb', 'cheb2leg', 'Leg2chebHaleTownsend',
           'Leg2Cheb', 'Cheb2Leg', 'FMMLeg2Cheb', 'FMMCheb2Leg']

Leg2Cheb = getattr(cython, 'Leg2Cheb', None)
Cheb2Leg = getattr(cython, 'Cheb2Leg', None)
Lambda = getattr(cython, 'Lambda', None)

class DLT:
    r"""Discrete Legendre Transform

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

        The scalar product is exactly like the forward transform, except that
        the Legendre mass matrix is not applied to the output.
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
        from . import fastgl
        if isinstance(axes, tuple):
            assert len(axes) == 1
            axis = axes[-1]
        elif isinstance(axes, int):
            axis = axes
        self.axis = axis
        assert kind in ('forward', 'scalar product', 'backward')
        self.kind = kind
        self.sl = slicedict(axis=axis, dimensions=input_array.ndim)
        self.si = islicedict(axis=axis, dimensions=input_array.ndim)
        N = self.N = input_array.shape[axis]
        xl, wl = fastgl.leggauss(N)
        xc = n_cheb.chebgauss(N)[0]
        thetal = np.arccos(xl)[::-1]
        thetac = np.arccos(xc)
        s = [None]*input_array.ndim
        s[axis] = slice(None)
        s = tuple(s) # broadcast to ndims
        self.dtheta = (thetal - thetac)[s]
        self.n = np.arange(N, dtype=float)[s]
        if kind in ('forward', 'scalar product'):
            self.wl = wl[s]
            self.nsign = np.ones(N)
            self.nsign[1::2] = -1
            if kind == 'forward': # apply inverse mass matrix as well
                self.nsign = self.nsign[s]*(self.n+0.5)
            else:
                self.nsign = self.nsign[s]
        ck = np.full(N, np.pi/2); ck[0] = np.pi
        self.ck = ck[s]
        U = input_array
        V = output_array if output_array is not None else U.copy()
        self.plan(U, V, kind, threads, flags)
        ##self.leg2chebclass = Leg2Cheb(U, axis=axis, maxs=100, use_direct=500)
        self.leg2chebclass = Leg2Cheb(U, domains=2, diagonals=16, axis=axis, maxs=100, use_direct=1000)

    def plan(self, U, V, kind, threads, flags):
        Uc = U.copy()
        Vc = V.copy()
        # dct and dst use the same input/output arrays
        if kind in ('forward', 'scalar product'):
            self.dct = DCT(Uc, axis=self.axis, type=2, threads=threads, flags=flags, output_array=Vc)
            self.dst = DST(Uc, axis=self.axis, type=2, threads=threads, flags=flags, output_array=Vc)
        else:
            self.dct = DCT(Uc, axis=self.axis, type=3, threads=threads, flags=flags, output_array=Vc)
            self.dst = DST(Uc, axis=self.axis, type=3, threads=threads, flags=flags, output_array=Vc)
        self._input_array = U
        self._output_array = V

    @property
    def input_array(self):
        return self._input_array

    @property
    def output_array(self):
        return self._output_array

    def __call__(self, input_array=None, output_array=None, **kw):
        if input_array is not None:
            self._input_array[:] = input_array

        # Set up x, which is input array to dct/dst
        # Some dcts apparently destroys the input_array, so need copy
        x = np.zeros_like(self.input_array)
        x[:] = self._input_array
        if self.kind in ('forward', 'scalar product'):
            x *= self.wl
        else:
            x = self.leg2chebclass(x.copy(), x)

        fk = self.dct(x).copy()
        nfac = 1
        y = 1
        n = 1
        converged = False
        while not converged:
            even = n % 2 == 0
            deven = (n+1)//2 % 2 == 0
            nfac *= n
            if self.kind == 'backward':
                x *= self.n
                y *= self.dtheta
            else:
                x *= self.dtheta
                y *= self.n
            sign = 1 if deven else -1
            fft = self.dct if even else self.dst
            h = fft(x)
            df = sign/nfac*y*h
            fk += df
            error = np.linalg.norm(df)
            #print(f"{n:4d} {error:2.4e}")
            converged = error < 1e-16
            n += 1

        if self.kind in ('forward', 'scalar product'):
            fk = self.leg2chebclass(fk.copy(), fk, transpose=True)
            fk *= self.nsign
        else:
            fk[:] = fk[self.sl[slice(-1, None, -1)]] # reverse
        if output_array is not None:
            output_array[:] = fk
            return output_array
        self._output_array[:] = fk
        return self._output_array

class DCT:
    """Discrete cosine transform with appropriate scaling

    input_array : array
    s : sequence of ints, optional
        Not used - included for compatibility with Numpy
    axes : sequence of ints, optional
        Axes over which to compute the real-to-real dct.
    type : int, optional
        Type of `dct <http://www.fftw.org/fftw3_doc/Real_002dto_002dReal-Transform-Kinds.html>`_

            - 1 - FFTW_REDFT00
            - 2 - FFTW_REDFT10,
            - 3 - FFTW_REDFT01,
            - 4 - FFTW_REDFT11
    threads : int, optional
        Number of threads used in computing dct.
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
    output_array : array, optional
        Array to be used as output array. Must be of correct shape, type,
        strides and alignment

    """
    def __init__(self, input_array, type=3, s=None, axis=-1, threads=1,
                 flags=(FFTW_MEASURE,), output_array=None):
        self.axis = axis
        self.type = type
        self.dct = fftw.dctn(input_array, axes=(axis,), type=type, threads=threads,
                             flags=flags, output_array=output_array)
        s = [slice(None)]*input_array.ndim
        s[self.axis] = slice(0, 1)
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
    """Discrete sine transform with appropriate scaling

    Parameters
    ----------
    input_array : array
    s : sequence of ints, optional
        Not used - included for compatibility with Numpy
    axes : sequence of ints, optional
        Axes over which to compute the real-to-real dst.
    type : int, optional
        Type of `dst <http://www.fftw.org/fftw3_doc/Real_002dto_002dReal-Transform-Kinds.html>`_

            - 1 - FFTW_RODFT00
            - 2 - FFTW_RODFT10
            - 3 - FFTW_RODFT01
            - 4 - FFTW_RODFT11
    threads : int, optional
        Number of threads used in computing dst.
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
    output_array : array, optional
        Array to be used as output array. Must be of correct shape, type,
        strides and alignment

    """
    def __init__(self, input_array, type=3, s=None, axis=-1, threads=1,
                 flags=None, output_array=None):
        self.axis = axis
        self.type = type
        self.dst = fftw.dstn(input_array, axes=(axis,), type=type, threads=threads,
                             flags=flags, output_array=output_array)
        self.sl = slicedict(axis=axis, dimensions=input_array.ndim)
        self.si = islicedict(axis=axis, dimensions=input_array.ndim)

    @property
    def input_array(self):
        return self.dst.input_array

    @property
    def output_array(self):
        return self.dst.output_array

    def destroy(self):
        self.dst.destroy()

    def __call__(self, input_array=None, output_array=None):
        if input_array is not None:
            self.input_array[:] = input_array
        x = self.input_array
        if self.type == 3:
            x[self.sl[slice(0, -1)]] = x[self.sl[slice(1, None)]]
            x[self.si[-1]] = 0
            out = self.dst()
        elif self.type == 2:
            out = self.dst()
            out[self.sl[slice(1, None)]] = out[self.sl[slice(0, -1)]]
            out[self.si[0]] = 0
        out /= 2
        if output_array is not None:
            output_array[:] = out
            return output_array
        return out

#Lambda = lambda z: np.exp(gammaln(z+0.5) - gammaln(z+1))

LxyE = lambda x, y: -0.5/(2*x+2*y+1)/(y-x)*Lambda(y-x-1)*Lambda(y+x-0.5)
MxyE = lambda x, y: Lambda(y-x)*Lambda(y+x)
Mxy = lambda x, y: Lambda((y-x)/2)*Lambda((y+x)/2)
Lxy = lambda x, y: -1/(x+y+1)/(y-x)*Lambda((y-x-2)/2)*Lambda((y+x-1)/2)
Hxy = lambda x, y: Lambda((y-x)/2)*Lambda((y+x)/2)/np.sqrt((x+y)*(y-x))

@runtimeoptimizer
def leg2cheb(cl, cc=None, axis=0, transpose=False):
    r"""Compute Chebyshev coefficients from Legendre coefficients

    .. math::

            \hat{c}^{cheb} = M \hat{c}^{leg}

    where :math:`\hat{c}^{cheb} \in \mathbb{R}^N` are the Chebyshev
    coefficients, :math:`\hat{c}^{leg} \in \mathbb{R}^N` the Legendre
    coefficients and :math:`M\in\mathbb{R}^{N \times N}` the matrix for the
    conversion. Note that if keyword transpose is true, then we compute

    .. math::

        \hat{a} = M^T \hat{b}

    for some vectors :math:`\hat{a}` and :math:`\hat{b}`.

    Paramters
    ---------
    cl : array
        The Legendre coefficients (if transpose is False)
    cc : array (return array), optional
        The Chebyshev coefficients (if transpose is False)
    axis : int, optional
        The axis over which to take the transform
    transpose : bool
        Whether to perform the transpose operation

    Note
    ----
    This is a low-memory implementation of the direct matrix vector
    product. The matrix :math:`M` is given by Alpert and Rokhlin 'A Fast Algorithm
    for the Evaluation of Legendre Expansions', SIAM Journal on Scientific
    and Statistical Computing, 12, 1, 158-179, (1991), 10.1137/0912009.
    The matrix is not explicitely created.
    """
    if cc is None:
        cc = np.zeros_like(cl)
    else:
        cc.fill(0)

    if axis > 0:
        cl = np.moveaxis(cl, axis, 0)
        cc = np.moveaxis(cc, axis, 0)
    N = cl.shape[0]
    k = np.arange(N)
    a = Lambda(k)
    sl = [None]*cc.ndim
    ck = np.full(N, np.pi/2); ck[0] = np.pi
    sl[0] = slice(None)
    ck = ck[tuple(sl)]
    if transpose is False:
        for n in range(0, N, 2):
            sl[0] = slice(n//2, N-n//2)
            cc[:(N-n)] += a[n//2]*a[tuple(sl)]*cl[n:]
        cc /= ck

    else:
        cx = cl/ck
        for n in range(0, N, 2):
            sl[0] = slice(n//2, N-n//2)
            cc[n:] += a[n//2]*a[tuple(sl)]*cx[:(N-n)]

    if axis > 0:
        cl = np.moveaxis(cl, 0, axis)
        cc = np.moveaxis(cc, 0, axis)
    return cc

@runtimeoptimizer
def cheb2leg(cc, cl=None, axis=0):
    r"""Compute Legendre coefficients from Chebyshev coefficients

    .. math::

            \hat{c}^{leg} = L \hat{c}^{cheb}

    where :math:`\hat{c}^{cheb} \in \mathbb{R}^N` are the Chebyshev
    coefficients, :math:`\hat{c}^{leg} \in \mathbb{R}^N` the Legendre
    coefficients and :math:`L\in\mathbb{R}^{N \times N}` the matrix for the
    conversion.

    Paramters
    ---------
    cc : array
        The Chebyshev coefficients
    cl : array
        The Legendre coefficients (return array)
    axis : int, optional
        The axis over which to take the transform

    Note
    ----
    This is a low-memory implementation of the direct matrix vector
    product. The matrix :math:`L` is given by Alpert and Rokhlin 'A Fast Algorithm
    for the Evaluation of Legendre Expansions', SIAM Journal on Scientific
    and Statistical Computing, 12, 1, 158-179, (1991), 10.1137/0912009.
    The matrix is not explicitely created.
    """
    if cl is None:
        cl = np.zeros_like(cc)
    else:
        cl.fill(0)
    if axis > 0:
        cl = np.moveaxis(cl, axis, 0)
        cc = np.moveaxis(cc, axis, 0)
    N = cc.shape[0]
    k = np.arange(N, dtype=float)
    k[0] = 1
    vn = cc*k
    a = 1/(2*Lambda(k)*k*(k+0.5))
    k[0] = 0
    a[0] = 2/np.sqrt(np.pi)
    cl[:] = np.sqrt(np.pi)*a*vn
    for n in range(2, N, 2):
        dn = Lambda(np.array([(n-2)/2]))/n
        cl[:(N-n)] -= dn*a[n//2:(N-n//2)]*vn[n:]
    cl *= (k+0.5)
    if axis > 0:
        cl = np.moveaxis(cl, 0, axis)
        cc = np.moveaxis(cc, 0, axis)
    return cl

class Leg2chebHaleTownsend:
    """Class for computing Chebyshev coefficients from Legendre coefficients

    Algorithm from::

        Nicholas Hale and Alex Townsend "A fast, simple and stable Chebyshev-
        Legendre transform using an asymptotic formula", SIAM J Sci Comput (2014)
        (https://epubs.siam.org/doi/pdf/10.1137/130932223)

    Parameters
    ----------
    input_array : array
        Legendre coefficients
    output_array : array
        The returned array
    axis : int
        The axis over which to perform the computation in case the input_array
        is multidimensional.
    nM : int
        Parameter, see Hale and Townsend (2014). Note that one must have N >> nM.
    Nmin : int
        Parameter. Choose direct matvec approach for N < Nmin
    """
    def __init__(self, input_array, output_array=None, axis=0, nM=50, Nmin=40000):
        self.axis = axis
        self.N = input_array.shape[axis]
        self.L = None
        self.T = None
        self.U = None
        self.a = None
        self.Nmin = Nmin
        self._input_array = input_array
        self._output_array = output_array if output_array is not None else input_array.copy()
        self.sl = slicedict(axis=axis, dimensions=input_array.ndim)
        if self.N > Nmin:
            from shenfun import config
            mod = config['optimization']['mode']
            self.lib = importlib.import_module('.'.join(('shenfun.optimization', mod, 'transforms')))
            N = self.N
            self.thetak = (np.arange(N)+0.5)*np.pi/N
            self.sintk = np.sin(self.thetak)
            alpha = self.alpha = min(1/np.log(N/nM), 0.5)
            if N < 1000:
                K = 1
            elif N < 10000:
                K = 2
            elif N < 1000000:
                K = 3
            else:
                K = 4
            self.K = K
            self.ix = {0: 0}
            for k in range(1, K+1):
                self.ix[k] = int((N + 1) / np.pi * np.arcsin(nM / N / alpha**k))

    @property
    def input_array(self):
        return self._input_array

    @property
    def output_array(self):
        return self._output_array

    def _Um(self, m):
        return np.sin((m+0.5)*(np.pi/2-self.thetak))/(2**(m+0.5)*self.sintk**(m-0.5))

    def _Vm(self, m):
        return np.cos((m+0.5)*(np.pi/2-self.thetak))/(2*self.sintk)**(m+0.5)

    @staticmethod
    def _Cn(n):
        return np.sqrt(4/np.pi)*np.exp(gammaln(n+1) - gammaln(n+3/2))

    def plan(self, input_array):
        from shenfun import FunctionSpace
        assert input_array.shape[self.axis] == self.N
        if self.T is not None:
            assert self.T.N == self.N
            return

        self.L = FunctionSpace(self.N, 'L')
        self.U = FunctionSpace(self.N, 'U', quad='GC')
        self.T = FunctionSpace(self.N, 'C', quad='GC')
        self.a = self.L.get_recursion_matrix(self.N+3, self.N+3).diags('dia').data

        if input_array.ndim > 1:
            self.T.plan(input_array.shape, self.axis, input_array.dtype, {})
            self.U.plan(input_array.shape, self.axis, input_array.dtype, {})

    def __call__(self, input_array=None, output_array=None, transpose=False):
        r"""Compute Chebyshev coefficients from Legendre. That is, compute

        .. math::

            \hat{c}^{cheb} = M \hat{c}^{leg}

        where :math:`\hat{c}^{cheb} \in \mathbb{R}^N` are the Chebyshev
        coefficients, :math:`\hat{c}^{leg} \in \mathbb{R}^N` the Legendre
        coefficients and :math:`M\in\mathbb{R}^{N \times N}` the matrix for the
        conversion. Note that if keyword 'transpose' is true, then we compute

        .. math::

            \hat{a} = M^T \hat{b}

        for some vectors :math:`\hat{a}` and :math:`\hat{b}`.

        The Chebyshev and Legendre coefficients are the regular coefficients to
        the series

        .. math::

            p_l(x) = \sum_{k=0}^{N} \hat{c}_k^{leg}L_k(x) \\
            p_c(x) = \sum_{k=0}^{N} \hat{c}_k^{cheb}T_k(x)

        and we get :math:`\{\hat{c}_k^{cheb}\}_{k=0}^N` by setting :math:`p_l(x)=p_c(x)`
        for :math:`x=\{x_i\}_{i=0}^N`, where :math:`x_i=\cos(i+0.5)\pi/N`.

        Parameters
        ----------
        input_array : array
        output_array : array
        transpose : bool
            Whether to compute the transpose of the regular transform

        Note
        ----
        For small N we use a direct method that costs approximately :math:`0.25 N^2`
        operations. For larger N (see 'Nmin' parameter) we use the fast routine of

            Hale and Townsend 'A fast, simple and stable Chebyshev-Legendre
            transform using an asymptotic formula', SIAM J Sci Comput (2014)

        """
        if input_array is not None:
            self._input_array[:] = input_array

        axis = self.axis
        N = self.N
        if N <= self.Nmin:
            out = leg2cheb(self.input_array, self.output_array, axis=axis, transpose=transpose)
            if output_array is not None:
                output_array[:] = out
                return output_array
            return out

        self.plan(self.input_array)
        sn = [None]*self.input_array.ndim
        sn[axis] = slice(None); sn = tuple(sn)
        hmn = np.ones(self.N)
        Tc = np.zeros_like(self.input_array)
        Uc = np.zeros_like(self.input_array)
        z = np.zeros_like(self.input_array)
        cn = self._Cn(np.arange(N))[sn]
        xi, wi = self.T.points_and_weights()

        if transpose is False:
            cn = self.input_array*cn
            na = np.arange(N)
            for m in range(10):
                if m > 0:
                    hmn *= (m-0.5)**2/(m*(na+m+0.5))
                cm = cn*hmn[sn]
                um = self._Um(m)[sn]
                vm = self._Vm(m)[sn]
                for k in range(1, self.K+1):
                    Tc[:] = 0
                    Uc[:] = 0
                    si = self.sl[slice(self.ix[k], N-self.ix[k])]
                    sk = self.sl[slice(int(self.alpha**k*N), int(self.alpha**(k-1)*N))]
                    Tc[sk] = cm[sk]
                    Uc[self.sl[slice(0, -1)]] = Tc[self.sl[slice(1, None)]]
                    z[si] += (vm*self.T.backward(Tc) + um*self.U.backward(Uc))[si]
            si = self.sl[slice(0, self.ix[1])]
            zx = np.zeros_like(z[si])
            self.lib.evaluate_expansion_all(self.input_array, zx, xi[si[axis]], self.axis, self.a) # recursive eval
            z[si] += zx
            si = self.sl[slice(N-self.ix[1], None)]
            self.lib.evaluate_expansion_all(self.input_array, zx, xi[si[axis]], self.axis, self.a)
            z[si] += zx
            for k in range(1, self.K):
                si = self.sl[slice(self.ix[k], self.ix[k+1])]
                sk = self.sl[slice(0, int(self.alpha**k*self.N))]
                zx = np.zeros_like(z[si])
                self.lib.evaluate_expansion_all(self.input_array[sk], zx, xi[si[axis]], axis, self.a)
                z[si] += zx
                si = self.sl[slice(self.N-self.ix[k+1], self.N-self.ix[k])]
                self.lib.evaluate_expansion_all(self.input_array[sk], zx, xi[si[axis]], axis, self.a)
                z[si] += zx
            si = self.sl[slice(self.ix[self.K], self.N-self.ix[self.K])]
            sk = self.sl[slice(0, int(self.alpha**(self.K)*self.N))]
            zx = np.zeros_like(z[si])
            self.lib.evaluate_expansion_all(self.input_array[sk], zx, xi[si[axis]], axis, self.a)
            z[si] += zx
            self._output_array = self.T.forward(z, self._output_array)

        else: # transpose
            ck = np.full(N, np.pi/2); ck[0] = np.pi
            ctilde = self.T.backward(self.input_array/ck[sn]).copy() # TN^{-T} * cl
            wu = self.U.points_and_weights()[1]
            U1 = np.zeros_like(self.input_array)
            for m in range(10):
                if m > 0:
                    hmn *= (m-0.5)**2/(m*(np.arange(N)+m+0.5))
                um = self._Um(m)[sn]
                vm = self._Vm(m)[sn]
                for k in range(1, self.K+1):
                    Tc[:] = 0
                    Uc[:] = 0
                    si = self.sl[slice(self.ix[k], self.N-self.ix[k])]
                    sk = self.sl[slice(int(self.alpha**k*self.N), int(self.alpha**(k-1)*self.N))]
                    Tc[si] = ctilde[si]
                    T0 = self.T.scalar_product(Tc*vm)
                    U0 = self.U.scalar_product(Tc/wu[sn]*wi[sn]*um)
                    U1[self.sl[slice(1, None)]] = U0[self.sl[slice(0, -1)]]
                    z[sk] += (cn*hmn[sn]*(T0+U1))[sk]
            sk = self.sl[slice(0, int(self.alpha**self.K*self.N))]
            zx = np.zeros_like(z[sk])
            z[sk] += restricted_product(self.L, wi[sn]*ctilde, zx, xi, 0, self.N, 0, axis, self.a)
            for k in range(self.K):
                sk = self.sl[slice(int(self.alpha**(k+1)*self.N), int(self.alpha**k*self.N))]
                zx = np.zeros_like(z[sk])
                z[sk] += restricted_product(self.L, wi[sn]*ctilde, zx, xi, 0, self.ix[k+1], sk[axis].start, axis, self.a)
                z[sk] += restricted_product(self.L, wi[sn]*ctilde, zx, xi, N-self.ix[k+1], N, sk[axis].start, axis, self.a)
            self._output_array[:] = z

        if output_array is not None:
            output_array[:] = self._output_array
            return output_array
        return self._output_array

@runtimeoptimizer
def restricted_product(L, input_array, output_array, xi, i0, i1, a0, axis, a):
    r"""Returns the restricted matrix vector product

    .. math::

        \sum_{i=i_0}^{i_1} \sum_{k=k_0}^{k_1} \hat{u}_k P_k(x_i)

    where the matrix :math:`P_k(x_i)` is the :math:`k`'th basis function of a
    family of orthogonal polynomials, at points :math:`\{x_i\}_{i_0}^{i_1}`.
    The array :math:`\boldsymbol{\hat{u}}` can be multidimensional, in which
    case the product is applied along the indicated axis.

    Parameters
    ----------
    L : instance of :class:`.SpectralBase`
    input_array : array
    output_array : array
    xi : Quadrature points for space L
    i0, i1 : integers
        Start and stop indices for xi
    a0 : int
        Start index for basis functions :math:`P_k`
    axis : int
        The axis of :math:`\hat{u}_k` to take the restricted product over in
        case :math:`\boldsymbol{\hat{u}}` is multidimensional
    a : array
        Recurrence coefficients for the orthogonal polynomials

    """
    N = xi.shape[0]
    Lnm = L.evaluate_basis(xi[i0:i1], i=a0)
    Ln = L.evaluate_basis(xi[i0:i1], i=a0+1)
    if a.shape[0] == 3:
        anm = a[0] # a_{n-1, n}
        ann = a[1] # a_{n, n}
        anp = a[2] # a_{n+1, n}
    else:
        anm = a[0]
        anp = a[1]
        ann = np.zeros(N+2)
    Lnp = (xi[i0:i1]-ann[1+a0])/anm[1+a0]*Ln - anp[1+a0]/anm[1+a0]*Lnm
    for k in range(len(output_array)):
        kp = k+a0
        s1 = 1/anm[kp+2]
        s2 = anp[kp+2]/anm[kp+2]
        a00 = ann[kp+2]
        s = 0.0
        for i in range(len(Ln)):
            s += Lnm[i]*input_array[i0+i]
            Lnm[i] = Ln[i]
            Ln[i] = Lnp[i]
            Lnp[i] = s1*(xi[i0+i]-a00)*Ln[i] - s2*Lnm[i]
        output_array[k] = s
    return output_array

def getChebyshev(level, D, s, diags, A, N, l2c=True):
    """Low-rank computation of Chebyshev coefficients

    Computes Chebyshev coefficients for all submatrices on a given
    level of a hierarchical decomposition.

    Parameters
    ----------
    level : int
        The level in the hierarchical decomposition
    D : int
        Domains size on level
    s : int
        Size of the smallest submatrices
    diags : int
        The number of neglected diagonals, that are treated using a
        direct approach
    A : list
        Each item is a flattened low-rank matrix of coefficients
    N : list
        Each item is the shape of the corresponding matrix item in A
    l2c : bool
        If True, the transform goes from Legendre to Chebyshev, and
        vice versa if False
    """
    from shenfun import FunctionSpace, TensorProductSpace
    h = s*get_h(level, D)
    i0, j0 = get_ij(level, 0, s, D, diags)
    Nb = get_number_of_blocks(level, D)
    T0 = FunctionSpace(100, 'C', domain=[j0+2*h, j0+4*h])
    fun = Mxy if l2c == True else Lxy
    w0 = fun(2*h-1, T0.mesh())
    m0 = T0.forward(w0)
    z = np.where(np.diff(abs(m0)) > 0)[0]
    Nx = 100 if len(z) < 2 else z[1]
    T0.domain = [j0+4*h, j0+6*h]
    w0 = fun(2*h-1, T0.mesh())
    m0 = T0.forward(w0)
    z = np.where(np.diff(abs(m0)) > 0)[0]
    Nx2 = 100 if len(z) < 2 else z[1]
    Nx2 = min(Nx2, Nx)
    #Nx, Nx2 = 50, 50
    T1 = FunctionSpace(Nx, 'C')
    S = TensorProductSpace(MPI.COMM_SELF, (T1, T1))
    f = np.zeros((Nx, Nx))
    dctn = fftw.dctn(np.zeros((Nx, Nx)), axes=(0, 1))
    xj = np.cos((np.arange(Nx)+0.5)*np.pi/Nx)
    xj2 = np.cos((np.arange(Nx2)+0.5)*np.pi/(Nx2))
    dctn2 = fftw.dctn(np.zeros((Nx2, Nx2)), axes=(0, 1))
    X = np.zeros((Nx, 1))
    Y = np.zeros((1, Nx))
    X2 = np.zeros((Nx2, 1))
    Y2 = np.zeros((1, Nx2))
    Mmax = 0
    for k in range(Nb):
        i0, j0 = get_ij(level, k, s, D, diags)
        for j in range(D[level]):
            S.bases[1].domain = 2*(j0+(j+1)*h), 2*(j0+(j+2)*h)
            for i in range(j+1):
                S.bases[0].domain = 2*(i0+i*h), 2*(i0+(i+1)*h)
                f[:] = 0
                if j > i:
                    Y2[0, :] = S.bases[1].map_true_domain(xj2)
                    X2[:, 0] = S.bases[0].map_true_domain(xj2)
                    f[:Nx2, :Nx2] = dctn2(fun(X2, Y2))/Nx2**2
                else:
                    Y[0, :] = S.bases[1].map_true_domain(xj)
                    X[:, 0] = S.bases[0].map_true_domain(xj)
                    f[:] = dctn(fun(X, Y))/Nx**2
                f[0] /= 2
                f[:, 0] /= 2
                z = np.where(np.diff(abs(f[0, 2:])) >= 0)[0]
                #z = np.where(abs(f[0]) < 1e-15)[0]
                Mmin = z[0]+2 if len(z) > 1 else Nx
                #Mmin = Nx
                A.append(f[:Mmin, :Mmin].ravel().copy())
                N.append(Mmin)
                Mmax = max(Mmax, Mmin)
    S.destroy()
    return Mmax

class FMMLevel:
    """Abstract base class for hierarchical matrix
    """
    def __init__(self, N, domains=None, levels=None, l2c=True, maxs=100, use_direct=-1):
        self.N = N
        self.use_direct = use_direct
        self._output_array = np.array([0])
        if N <= use_direct:
            self.Nn = N
            return

        if domains is not None:
            if isinstance(domains, int):
                if levels is None:
                    doms = np.cumsum(domains**np.arange(16))
                    levels = np.where(N/(2*(1+doms)) <= maxs)[0][0]
                levels = max(1, levels)
                self.D = np.full(levels, domains, dtype=int)
            else:
                domains = np.atleast_1d(domains)
                levels = len(domains)
                self.D = domains
        else:
            if levels is None:
                domains = 2
                doms = np.cumsum(domains**np.arange(20))
                levels = np.where(N/(2*(1+doms)) <= maxs)[0][0]
                levels = max(1, levels)
            else:
                for domains in range(2, 100):
                    Nb = np.cumsum(domains**np.arange(levels+1))[levels]
                    s = N//(2*(1+Nb))
                    if s <= maxs:
                        break
            self.D = np.full(levels, domains, dtype=int)

        self.L = max(1, levels)
        Nb = get_number_of_blocks(self.L, self.D)
        #self.diags = diagonals
        self.axis = 0
        # Create new (even) N with minimal size according to diags and N
        # Pad input array with zeros
        #s, rest = divmod(N//2-diagonals, Nb)
        s = np.ceil(N/(2*(1+Nb))).astype(int)
        Nn = 2*s*(1+Nb)
        #Nn = N
        #if rest > 0 or Nn%2 == 1:
        #    s += 1
        #    Nn = 2*(diagonals+s*Nb)

        self.Nn = Nn
        self.s = s
        self.diags = s
        fk = []
        Nk = []
        self.Mmin = np.zeros(self.L, dtype=int)
        for level in range(self.L-1, -1, -1):
            Nx = getChebyshev(level, self.D, s, s, fk, Nk, l2c)
            self.Mmin[level] = Nx
        self.fk = np.hstack(fk)
        self.Nk = np.array(Nk, dtype=int)
        Mmax = self.Mmin.max()
        TT = {}
        for level in range(self.L-1, -1, -1):
            if self.D[level] not in TT:
                TT[self.D[level]] = conversionmatrix(self.D[level], Mmax)
        self.Th = np.concatenate([TT[d] for d in np.unique(self.D)])
        self.ThT = np.array([copy(self.Th[i].transpose()) for i in range(self.Th.shape[0])])
        self.T = np.zeros((2, s, self.Mmin[-1]))
        Ti = n_cheb.chebvander(np.linspace(-1, 1, 2*s+1)[:-1], self.Mmin[-1]-1)
        self.T[0] = Ti[::2]
        self.T[1] = Ti[1::2]

    def get_M_shapes(self):
        M = []
        for level in range(self.L):
            Nb = get_number_of_blocks(level, self.D)
            M.append([np.sum((self.Nk[level][1:])**2), Nb*self.D[level]*(self.D[level]+1)//2])
        return M

    def plan(self, shape, dtype, axis, output_array=None, use_direct=-1):
        if self._output_array.shape == shape and axis == self.axis and self._output_array.dtype == dtype:
            return
        self.axis = axis
        self.sl = slicedict(axis=axis, dimensions=len(shape))
        self.si = islicedict(axis=axis, dimensions=len(shape))
        if shape[axis] > use_direct:
            oddevenshape = list(shape)
            oddevenshape[axis] = self.Nn//2
            self.cont_input_array = np.zeros(oddevenshape, dtype=dtype)
            self.cont_output_array = np.zeros(oddevenshape, dtype=dtype)
        self._output_array = np.zeros(shape, dtype=dtype) if output_array is None else output_array

    def plotrank(self):
        import matplotlib.pyplot as plt
        z = np.zeros((self.Nn, self.Nn))
        ik = 0
        for level in range(self.L-1, -1, -1):
            for block in range(get_number_of_blocks(level, self.D)):
                h = 2 * self.s * get_h(level, self.D)
                i0, j0 = get_ij(level, block, self.s, self.D, self.diags)
                for j in range(self.D[level]):
                    for i in range(j + 1):
                        z[2*i0+i*h:2*i0+(i+1)*h, 2*j0+(j+1)*h:2*j0+(j+2)*h] = self.Nk[ik]
                        ik += 1
        plt.figure()
        plt.imshow(z, cmap='gray')
        plt.title('Submatrix rank')
        plt.colorbar()
        plt.show()

def get_ij(level, block, s, D, diagonals):
    i0 = block*D[level]*s*get_h(level, D)
    j0 = diagonals+i0
    for i in range(level+1, len(D)):
        j0 += s*get_h(i, D)
    return (i0, j0)

def get_h(level, D):
    #return (self.domains-1)**(self.L-1-level) # const D
    return np.prod(D[level+1:])

def get_number_of_blocks(level, D):
    #return np.sum(D[0]**np.arange(level+1))   # const D
    return 1+np.sum(np.cumprod(np.flip(D[:level])))

def get_number_of_submatrices(D):
    Ns = 0
    for level in range(len(D)):
        Ns += get_number_of_blocks(level, D)*D[level]*(D[level]+1)//2
    return Ns

class FMMLeg2Cheb(FMMLevel):
    """Transform Legendre coefficients to Chebyshev coefficients

    Parameters
    ----------
    input : int or input array
        The length of the array to transform or the array itself
    diagonals : int
        The number of neglected diagonals, that are treated using a
        direct approach
    domains : None, int or sequence of ints
        The domain sizes for all levels

        If domains=None then an appropriate domain size is computed
        according to the given number of levels and maxs. If the
        number of levels is also None, then domains is set to 2
        and the number of levels is computed from maxs.

        If domains is an integer, then this integer is used for each
        level, and the number of levels is either given or computed
        according to maxs

        If domains is a sequence of integers, then these are the
        domain sizes for all the levels and the length of this
        sequence is the number of levels.

    levels : None or int
        The number of levels in the hierarchical matrix

        If levels is None, then it is computed according to domains
        and maxs

    l2c : bool
        If True, the transform goes from Legendre to Chebyshev, and
        vice versa if False

    maxs : int
        The maximum size of the smallest submatrices (on the highest
        level). This number is used if the number of levels or domain
        sizes need to be computed.

    axis : int
        The axis over which to apply the transform if the input array
        is a multidimensional array.

    use_direct : int
        Use direct method if N is smaller than this number

    """
    def __init__(self, input, output_array=None, domains=None, levels=None, maxs=100, axis=0, use_direct=-1):
        if isinstance(input, int):
            N = input
            shape = (N,)
            dtype = float
        elif isinstance(input, np.ndarray):
            N = input.shape[axis]
            shape = input.shape
            dtype = input.dtype
        FMMLevel.__init__(self, N, domains=domains, levels=levels, maxs=maxs, use_direct=use_direct)
        self.a = Lambda(np.arange(self.Nn, dtype=float))
        self.plan(shape, dtype, axis, output_array, use_direct)

    def __call__(self, input_array, output_array=None, transpose=False):
        """Execute transform

        Parameters
        ----------
        input_array : array
            The array to be transformed
        output_array : None or array
            The return array. Will be created if None.
        transpose : bool
            If True, then apply the transpose operation :math:`A^Tu`
            instead of the default :math:`Au`, where :math:`A` is the
            matrix and :math:`u` is the array of Legendre coefficients.

        """
        self.plan(input_array.shape, input_array.dtype, self.axis, output_array, self.use_direct)
        if input_array.shape[self.axis] <= self.use_direct and input_array.ndim == 1:
            self._output_array[...] = 0
            _leg2cheb(input_array, self._output_array, self.a, transpose)
            if output_array is not None:
                output_array[...] = self._output_array
                return output_array
            return self._output_array
        if transpose is True:
            input_array = input_array*2/np.pi # makes copy (preserve input)
            input_array[self.si[0]] *= 0.5
        if input_array.shape[self.axis] <= self.use_direct:
            self._output_array.fill(0)
            FMMdirect1(input_array, self._output_array, self.axis, self.a, self.N//2, transpose)
        else:
            self.apply(input_array, self._output_array, transpose)
        if transpose is False:
            self._output_array *= (2/np.pi)
            self._output_array[self.si[0]] *= 0.5
        return self._output_array

    def apply(self, input_array, output_array, transpose):
        FMMcheb(input_array, output_array, self.axis, self.Nn, self.fk, self.Nk, self.T, self.Th, self.ThT, self.D, self.Mmin, self.s, self.diags, transpose)
        FMMdirect1(input_array, output_array, self.axis, self.a, self.diags, transpose)
        FMMdirect2(input_array, output_array, self.axis, self.a, 2*self.s, get_number_of_blocks(self.L, self.D), 2*self.diags, transpose)


class FMMCheb2Leg(FMMLevel):
    """Transform Chebyshev coefficients to Legendre coefficients

    Parameters
    ----------
    input : int or input array
        The length of the array to transform or the array itself
    diagonals : int
        The number of neglected diagonals, that are treated using a
        direct approach
    domains : None, int or sequence of ints
        The domain sizes for all levels

        If domains=None then an appropriate domain size is computed
        according to the given number of levels and maxs. If the
        number of levels is also None, then domains is set to 2
        and the number of levels is computed from maxs.

        If domains is an integer, then this integer is used for each
        level, and the number of levels is either given or computed
        according to maxs

        If domains is a sequence of integers, then these are the
        domain sizes for all the levels and the length of this
        sequence is the number of levels.

    levels : None or int
        The number of levels in the hierarchical matrix

        If levels is None, then it is computed according to domains
        and maxs

    l2c : bool
        If True, the transform goes from Legendre to Chebyshev, and
        vice versa if False

    maxs : int
        The maximum size of the smallest submatrices (on the highest
        level). This number is used if the number of levels or domain
        sizes need to be computed.

    axis : int
        The axis over which to apply the transform if the input array
        is a multidimensional array.

    use_direct : int
        Use direct method if N is smaller than this number
    """
    def __init__(self, input, output_array=None, diagonals=16, domains=None, levels=None, maxs=100, axis=0, use_direct=-1):
        if isinstance(input, int):
            N = input
            shape = (N,)
            dtype = float
        elif isinstance(input, np.ndarray):
            N = input.shape[axis]
            shape = input.shape
            dtype = input.dtype
        FMMLevel.__init__(self, N, domains=domains,
                          diagonals=diagonals, levels=levels, maxs=maxs, l2c=False, use_direct=use_direct)
        k = np.arange(self.Nn, dtype='d')
        k[0] = 1
        self.dn = Lambda((k[::2]-2)/2)/k[::2]
        self.a = 1/(2*Lambda(k)*k*(k+0.5))
        self.a[0] = 2/np.sqrt(np.pi)
        self.plan(shape, dtype, axis, output_array, use_direct)

    def __call__(self, input_array, output_array=None):
        """Execute transform

        Parameters
        ----------
        input_array : array
            The array to be transformed
        output_array : None or array
            The return array. Will be created if None.

        """
        assert isinstance(input_array, np.ndarray)
        self.plan(input_array.shape, input_array.dtype, self.axis, output_array, self.use_direct)
        if input_array.shape[self.axis] <= self.use_direct and input_array.ndim == 1:
            self._output_array[...] = 0
            _cheb2leg(input_array, self._output_array, self.dn, self.a)
            if output_array is not None:
                output_array[...] = self._output_array
                return output_array
            return self._output_array

        self.sl = sl = slicedict(axis=self.axis, dimensions=input_array.ndim)
        si = [None]*input_array.ndim
        si[self.axis] = slice(None)
        w0 = input_array.copy()
        w0[sl[slice(1, self.N)]] *= np.arange(1, self.N)[tuple(si)]

        if input_array.shape[self.axis] <= self.use_direct:
            self._output_array.fill(0)
            FMMdirect4(w0, self._output_array, self.axis, self.dn[:self.N//2], self.a[:self.N], self.N//2)
        else:
            self.apply(w0, self._output_array)

        self._output_array *= (np.arange(self.N)+0.5)[tuple(si)]
        if output_array is not None:
            output_array[...] = self._output_array
            return output_array
        return self._output_array

    def apply(self, input_array, output_array):
        FMMcheb(input_array, output_array, self.axis, self.Nn, self.fk, self.Nk, self.T, self.Th, self.ThT, self.D, self.Mmin, self.s, self.diags, False)
        FMMdirect4(input_array, output_array, self.axis, self.dn[:self.N//2], self.a[:self.N], self.diags)
        FMMdirect3(input_array, output_array, self.axis, self.dn, self.a, 2*self.s, get_number_of_blocks(self.L, self.D), 2*self.diags)

@runtimeoptimizer
def _cheb2leg(u, v, dn, a):
    pass

@runtimeoptimizer
def _leg2cheb(u, v, a, trans):
    pass

@runtimeoptimizer
def FMMdirect1(u, v, axis, a, n0, trans):
    N = u.shape[0]
    if trans is False:
        for n in range(0, n0):
            v[:(N-2*n)] += a[n]*a[n:(N-n)]*u[2*n:]
    else:
        for n in range(0, n0):
            v[2*n:] += a[n]*a[n:(N-n)]*u[:(N-2*n)]

@runtimeoptimizer
def FMMdirect2(u, v, axis, a, h, Nb, n0, trans):
    N = v.shape[0]
    for k in range(Nb):
        i0 = k*h
        j0 = n0+i0
        for n in range(0, h, 2):
            a0 = i0+(n0+n)//2
            Nm = min(N-(j0+n), h-n)
            if trans is True:
                v[j0+n:(j0+Nm)] += a[(n+n0)//2]*a[a0:a0+Nm]*u[i0:i0+Nm]
            else:
                v[i0:(i0+Nm)] += a[(n+n0)//2]*a[a0:a0+Nm]*u[j0+n:j0+n+Nm]

#from shenfun import optimization
#FMMdirect2 = optimization.numba.transforms.FMMdirect2
#FMMdirect3 = optimization.numba.transforms.FMMdirect3

@runtimeoptimizer
def FMMdirect3(u, v, axis, dn, a, h, Nb, n0):
    N = v.shape[0]
    for k in range(Nb):
        i0 = k*h
        j0 = n0+i0
        for n in range(0, h, 2):
            a0 = i0+(n0+n)//2
            Nm = min(N-(j0+n), h-n)
            v[i0:(i0+Nm)] -= dn[(n+n0)//2]*a[a0:a0+Nm]*u[j0+n:j0+n+Nm]

@runtimeoptimizer
def FMMdirect4(u, v, axis, dn, a, n0):
    N = u.shape[0]
    v[:] += np.sqrt(np.pi)*a*u
    for n in range(1, n0):
        v[:(N-2*n)] -= dn[n]*a[n:(N-n)]*u[2*n:]

@runtimeoptimizer
def FMMcheb(input_array, output_array, axis, Nn, fk, Nk, T, Th, ThT, D, Mmin, s, diags, transpose):
    assert input_array.ndim == 1, 'Use Cython for multidimensional'
    L = len(D)
    Mmax = max(Mmin)
    N = input_array.shape[0]
    cia = np.zeros(Nn//2, dtype=input_array.dtype)
    coa = np.zeros(Nn//2, dtype=input_array.dtype)
    uD = np.hstack((0, np.unique(D)))
    output_array[:] = 0
    wk = [None]*L
    ck = [None]*L
    for level in range(L):
        wk[level] = np.zeros((get_number_of_blocks(level, D), D[level], Mmax))
        ck[level] = np.zeros((get_number_of_blocks(level, D), D[level], Mmax))
    for sigma in (0, 1):
        cia[:] = 0
        coa[:] = 0
        cia[:N//2+(N%2)*(1-sigma)] = input_array[sigma::2]
        if sigma == 1:
            for level in range(L):
                wk[level][:] = 0
                ck[level][:] = 0

        ik = 0
        Nc = 0
        if transpose is False:
            for level in range(L-1, -1, -1):
                h = s*get_h(level, D)
                M0 = Mmin[level]
                s0 = np.searchsorted(uD, D[level])-1
                s1 = np.cumsum(uD)[s0]
                TT = Th[s1:s1+D[level]]
                #if level == L-1:
                #    wk[level][...] = np.dot(cia[diags+s:].reshape((D[L-1]*get_number_of_blocks(L-1, D), s)), T[sigma]).reshape((get_number_of_blocks(L-1, D), D[L-1], Mmax))

                for block in range(get_number_of_blocks(level, D)):
                    i0, j0 = get_ij(level, block, s, D, diags)
                    b0, q0 = divmod(block-1, D[level-1])
                    for q in range(D[level]):
                        if level == L-1:
                            wk[level][block, q, :M0] = np.dot(cia[j0+(q+1)*h:j0+(q+2)*h], T[sigma, :, :M0])
                        if level > 0 and block > 0:
                            wk[level-1][b0, q0, :M0] += np.dot(TT[q, :M0, :M0], wk[level][block, q, :M0])
                        for p in range(q+1):
                            M = Nk[ik]
                            ck[level][block, p, :M] += np.dot(fk[Nc:(Nc+M*M)].reshape(M, M), wk[level][block, q, :M])
                            Nc += M*M
                            ik += 1

            for level in range(L):
                M0 = Mmin[level]
                if level < L-1:
                    s0 = np.searchsorted(uD, D[level+1])-1
                    s1 = np.cumsum(uD)[s0]
                    TT = ThT[s1:s1+D[level+1]]
                    for block in range(get_number_of_blocks(level+1, D)-1):
                        b0, p0 = divmod(block, D[level])
                        for p in range(D[level+1]):
                            ck[level+1][block, p, :M0] += np.dot(TT[p, :M0, :M0], ck[level][b0, p0, :M0])

                else:
                    for block in range(get_number_of_blocks(level, D)):
                        i0, j0 = get_ij(level, block, s, D, diags)
                        for p in range(D[level]):
                            coa[i0+p*s:i0+(p+1)*s] = np.dot(T[sigma], ck[level][block, p])

        else:
            for level in range(L-1, -1, -1):
                h = s*get_h(level, D)
                M0 = Mmin[level]
                s0 = np.searchsorted(uD, D[level])-1
                s1 = np.cumsum(uD)[s0]

                for block in range(get_number_of_blocks(level, D)):
                    j0, i0 = get_ij(level, block, s, D, diags)
                    b0, q0 = divmod(block, D[level-1])
                    TT = Th[s1:s1+D[level]]
                    for p in range(D[level]):
                        for q in range(p+1):
                            if q == p:
                                if level == L-1:
                                    wk[level][block, q, :M0] = np.dot(cia[j0+q*h:j0+(q+1)*h], T[sigma, :, :M0])
                                if level > 0 and block < get_number_of_blocks(level, D)-1:
                                    wk[level-1][b0, q0, :M0] += np.dot(TT[q, :M0, :M0], wk[level][block, q, :M0])
                            M = Nk[ik]
                            ck[level][block, p, :M] += np.dot(wk[level][block, q, :M], fk[Nc:(Nc+M*M)].reshape(M, M))
                            Nc += M*M
                            ik += 1

            for level in range(L):
                M0 = Mmin[level]
                if level < L-1:
                    s0 = np.searchsorted(uD, D[level+1])-1
                    s1 = np.cumsum(uD)[s0]
                    TT = Th[s1:s1+D[level+1]]
                    for block in range(1, get_number_of_blocks(level+1, D)):
                        for p in range(D[level+1]):
                            b0, p0 = divmod(block-1, D[level])
                            ck[level+1][block, p, :M0] += np.dot(ck[level][b0, p0, :M0], TT[p, :M0, :M0])

                else:
                    for block in range(get_number_of_blocks(level, D)):
                        j0, i0 = get_ij(level, block, s, D, diags)
                        for p in range(D[level]):
                            coa[i0+(p+1)*s:i0+(p+2)*s] = np.dot(T[sigma], ck[level][block, p])

        output_array[sigma::2] = coa[:N//2+(N%2)*(1-sigma)]

def conversionmatrix(D : int, M : int) -> np.ndarray:
    from mpi4py_fft.fftw import dctn
    T = np.zeros((D, M, M))
    k = np.arange(M)
    X = np.cos((k+0.5)*np.pi/M)[None, :]
    dct = dctn(np.zeros((M, M)), axes=(1,))
    for q in range(D):
        T[q] = dct(np.cos(k[:, None]*np.arccos((X+1+2*q-D)/D)))/M
        T[q, :, 0] /= 2
    return T
