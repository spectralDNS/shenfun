import importlib
from scipy.special import gammaln
import numpy as np
from numpy.polynomial import chebyshev as n_cheb
from mpi4py import MPI
from mpi4py_fft import fftw
from mpi4py_fft.fftw.utilities import FFTW_MEASURE, FFTW_PRESERVE_INPUT
from shenfun.optimization import runtimeoptimizer
from . import fastgl

__all__ = ['DLT', 'leg2cheb', 'cheb2leg', 'Leg2cheb']

comm = MPI.COMM_WORLD

class DLT:
    """Discrete Legendre Transform

    A class for performing fast FFT-based discrete Legendre transforms, both
    forwards and backwards. Based on::

        Nicholas Hale and Alex Townsend "A fast FFT-based discrete Legendre
        transform", IMA Journal of Numerical Analysis (2015)
        (https://arxiv.org/abs/1505.00354)

        Nicholas Hale and Alex Townsend "A fast, simple and stable Chebyshev-
        Legendre transform using an asymptotic formula", SIAM J Sci Comput (2014)
        (https://epubs.siam.org/doi/pdf/10.1137/130932223)

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

        The scalar product is exactly like the forward transform, except that the
        Legendre mass matrix is not applied to the output.
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
        xl, wl = fastgl.leggauss(N)
        xc = n_cheb.chebgauss(N)[0]
        thetal = np.arccos(xl)[::-1]
        thetac = np.arccos(xc)
        s = [None]*input_array.ndim
        s[axis] = slice(None)
        self.dtheta = (thetal - thetac)[tuple(s)]
        self.n = np.arange(N, dtype=float)[tuple(s)]
        s0 = [slice(None)]*input_array.ndim
        s0[axis] = slice(-1, None, -1) # reverse
        self.s0 = tuple(s0)
        if kind in ('forward', 'scalar product'):
            self.wl = wl[tuple(s)]
            self.nsign = np.ones(N)
            self.nsign[1::2] = -1
            if kind == 'forward': # apply inverse mass matrix as well
                self.nsign = self.nsign[tuple(s)]*(self.n+0.5)
            else:
                self.nsign = self.nsign[tuple(s)]
        ck = np.full(N, np.pi/2); ck[0] = np.pi
        self.ck = ck[tuple(s)]
        U = input_array
        V = output_array if output_array is not None else U.copy()
        self.plan(U, V, kind, threads, flags)
        self.leg2chebclass = Leg2cheb(U, axis=axis)

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
        x = np.zeros_like(self.dct.input_array)
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
            df = sign/nfac*y*fft(x)
            fk += df
            error = np.linalg.norm(df)
            #print(f"{n:4d} {error:2.4e}")
            converged = error < 1e-14
            n += 1

        if self.kind in ('forward', 'scalar product'):
            fk = self.leg2chebclass(fk.copy(), fk, transpose=True)
            fk *= self.nsign
        else:
            fk[:] = fk[self.s0] # reverse
        if output_array is not None:
            output_array[:] = fk
            return output_array
        self._output_array[:] = fk
        return self._output_array

class DCT:
    """Discrete cosine transform

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
        s0 = [slice(None)]*input_array.ndim
        s[self.axis] = 0
        s0[self.axis] = None
        self.s = tuple(s)
        self.s0 = tuple(s0)

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
            out += self.dct.input_array[self.s][self.s0]
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

Omega = lambda z: np.exp(gammaln(z+0.5) - gammaln(z+1))

@runtimeoptimizer
def leg2cheb(cl, cc, axis=0, transpose=False):
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
    cc : array (return array)
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
    if axis > 0:
        cl = np.moveaxis(cl, axis, 0)
        cc = np.moveaxis(cc, axis, 0)
    cc.fill(0)
    N = cl.shape[0]
    k = np.arange(N)
    a = Omega(k)
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
def cheb2leg(cc, cl, axis=0):
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
    if axis > 0:
        cl = np.moveaxis(cl, axis, 0)
        cc = np.moveaxis(cc, axis, 0)
    N = len(cc)
    k = np.arange(N)
    k[0] = 1
    vn = cc*k
    a = 1/(2*Omega(k)*k*(k+0.5))
    k[0] = 0
    a[0] = 2/np.sqrt(np.pi)
    cl[:] = np.sqrt(np.pi)*a*vn
    for n in range(2, N, 2):
        dn = Omega((n-2)/2)/n
        cl[:(N-n)] -= dn*a[n//2:(N-n//2)]*vn[n:]
    cl *= (k+0.5)
    if axis > 0:
        cl = np.moveaxis(cl, 0, axis)
        cc = np.moveaxis(cc, 0, axis)
    return cl

class Leg2cheb:
    """Class for computing Chebyshev coefficients from Legendre coefficients

    Parameters
    ----------
    input_array : array
        Legendre coefficients
    axis : int
        The axis over which to perform the computation in case the input_array
        is multidimensional.
    output_array : array
        The returned array
    nM : int
        Parameter, see Hale and Townsend (2014). Note that one must have N >> nM.
    Nmin : int
        Parameter. Choose direct matvec approach for N < Nmin
    """
    def __init__(self, input_array, axis=0, output_array=None, nM=50, Nmin=400):
        self.axis = axis
        self.N = input_array.shape[axis]
        self.L = None
        self.T = None
        self.U = None
        self.a = None
        self.Nmin = Nmin
        self._input_array = input_array
        self._output_array = output_array if output_array is not None else input_array.copy()
        if self.N > Nmin:
            from shenfun import config
            mod = config['optimization']['mode']
            self.lib = importlib.import_module('.'.join(('shenfun.optimization', mod, 'transforms')))
            N = self.N
            self.thetak = (np.arange(N)+0.5)*np.pi/N
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
        return np.sin((m+0.5)*(np.pi/2-self.thetak))*np.sin(self.thetak)/(2*np.sin(self.thetak))**(m+0.5)

    def _Vm(self, m):
        return np.cos((m+0.5)*(np.pi/2-self.thetak))/(2*np.sin(self.thetak))**(m+0.5)

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
        self.T = FunctionSpace(self.N, 'C', quad='GC')
        self.U = FunctionSpace(self.N, 'U', quad='GC')
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
        conversion. Note that if keyword transpose is true, then we compute

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
        operations. For larger N (>200) we use the fast routine of

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
        si = [slice(None)]*self.input_array.ndim
        sk = [slice(None)]*self.input_array.ndim
        sm = [slice(None)]*self.input_array.ndim
        sp = [slice(None)]*self.input_array.ndim
        sn = [None]*self.input_array.ndim
        sn[axis] = slice(None); sn = tuple(sn)
        sm[axis] = slice(0, -1); sm = tuple(sm)
        sp[axis] = slice(1, None); sp = tuple(sp)
        hmn = np.ones(self.N)
        Tc = np.zeros_like(self.input_array)
        Uc = np.zeros_like(self.input_array)
        z = np.zeros_like(self.input_array)
        cn = self._Cn(np.arange(N))[sn]
        xi, wi = self.T.points_and_weights()

        if transpose is False:
            cn = self.input_array*cn
            for m in range(10):
                if m > 0:
                    hmn *= (m-0.5)**2/(m*(np.arange(N)+m+0.5))
                cm = cn*hmn[sn]
                um = self._Um(m)[sn]
                vm = self._Vm(m)[sn]
                for k in range(1, self.K+1):
                    Tc[:] = 0
                    Uc[:] = 0
                    si[axis] = slice(int(self.alpha**k*N), int(self.alpha**(k-1)*N))
                    Tc[tuple(si)] = cm[tuple(si)]
                    Uc[sm] = Tc[sp]
                    si[axis] = slice(self.ix[k], N-self.ix[k])
                    z[tuple(si)] += (vm*self.T.backward(Tc) + um*self.U.backward(Uc))[tuple(si)]
            si[axis] = slice(0, self.ix[1])
            self.lib.evaluate_expansion_all(self.input_array, z[tuple(si)], xi[si[axis]], self.axis, self.a) # recursive eval
            si[axis] = slice(N-self.ix[1], None)
            self.lib.evaluate_expansion_all(self.input_array, z[tuple(si)], xi[si[axis]], self.axis, self.a)
            for k in range(1, self.K):
                si[axis] = slice(self.ix[k], self.ix[k+1])
                sk[axis] = slice(0, int(self.alpha**k*self.N))
                zx = np.zeros_like(z[tuple(si)])
                self.lib.evaluate_expansion_all(self.input_array[tuple(sk)], zx, xi[si[axis]], self.axis, self.a)
                z[tuple(si)] += zx
                self.lib.evaluate_expansion_all(self.input_array[tuple(sk)], zx, xi[si[axis]], self.axis, self.a)
                si[axis] = slice(self.N-self.ix[k+1], self.N-self.ix[k])
                z[tuple(si)] += zx
            si[axis] = slice(self.ix[self.K], self.N-self.ix[self.K])
            zx = np.zeros_like(z[tuple(si)])
            sk[axis] = slice(0, int(self.alpha**(self.K)*self.N))
            self.lib.evaluate_expansion_all(self.input_array[tuple(sk)], zx, xi[si[axis]], self.axis, self.a)
            z[tuple(si)] += zx
            self._output_array = self.T.forward(z, self._output_array)

        else: # transpose
            ck = np.full(N, np.pi/2); ck[0] = np.pi
            ctilde = self.T.backward(self.input_array/ck[sn]).copy() # TN^{-T} * cl
            wu = self.U.points_and_weights()[1]
            U1 = np.zeros_like(self.input_array)
            for m in range(10):
                if m > 0:
                    hmn *= (m-0.5)**2/(m*(np.arange(N)+m+0.5))
                um = self.Um(m)[sn]
                vm = self.Vm(m)[sn]
                for k in range(1, self.K+1):
                    Tc[:] = 0
                    Uc[:] = 0
                    si[axis] = slice(self.ix[k], self.N-self.ix[k])
                    sk[axis] = slice(int(self.alpha**k*self.N), int(self.alpha**(k-1)*self.N))
                    Tc[tuple(si)] = ctilde[tuple(si)]
                    T0 = self.T.scalar_product(Tc*vm)
                    U0 = self.U.scalar_product(Tc/wu[sn]*wi[sn]*um)
                    U1[sp] = U0[sm]
                    z[tuple(sk)] += (cn*hmn[sn]*(T0+U1))[tuple(sk)]
            sk[axis] = slice(0, int(self.alpha**self.K*self.N))
            restricted_product(self.L, wi[sn]*ctilde, z[tuple(sk)], xi, 0, self.N, 0, axis, self.a)
            for k in range(self.K):
                sk[axis] = slice(int(self.alpha**(k+1)*self.N), int(self.alpha**k*self.N))
                zx = np.zeros_like(z[tuple(sk)])
                z[tuple(sk)] += restricted_product(self.L, wi[sn]*ctilde, zx, xi, 0, self.ix[k+1], sk[axis].start, axis, self.a)
                z[tuple(sk)] += restricted_product(self.L, wi[sn]*ctilde, zx, xi, N-self.ix[k+1], N, sk[axis].start, axis, self.a)
            self._output_array[:] = z

        if output_array is not None:
            output_array[:] = self._output_array
            return output_array
        return self._output_array

@runtimeoptimizer
def restricted_product(L, input_array, output_array, xi, i0, i1, a0, axis, a):
    r"""Returns the restricted product matrix vector product

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
