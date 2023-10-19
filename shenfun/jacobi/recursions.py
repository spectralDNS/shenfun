"""
Recursions for Jacobi polynomials, or standardized Jacobi polynomials

"""
from copy import deepcopy
import numpy as np
import sympy as sp
from shenfun.matrixbase import SparseMatrix

x = sp.Symbol('x', real=True)
delta = sp.KroneckerDelta
half = sp.Rational(1, 2)
m, n, k = sp.symbols('m,n,k', real=True, integer=True, positive=True)
alfa, beta = sp.symbols('a,b', real=True)

def jt(alf, bet, n):
    """Jacobi polynomial

    Parameters
    ----------
    alf, bet : numbers
        Jacobi parameters
    n : int
        Index
    """
    return sp.jacobi(n, alf, bet, x)

def djt(alf, bet, n, k):
    """k'th derivative of Jacobi polynomial

    Parameters
    ----------
    alf, bet : numbers
        Jacobi parameters
    n : int
        Index
    k : int
        The k'th derivative
    """
    return sp.diff(jt(alf, bet, n), x, k)

def cn(alf, bet, n):
    r"""Scaling function

    .. math::

        c_n^{(\alpha, \beta)} = \frac{1}{P^{(\alpha,\beta)}_n(1)}

    Parameters
    ----------
    alf, bet : numbers
        Jacobi parameters
    n : int
        Index

    Note
    ----
    Used by Legendre and Chebyshev polynomials of first kind.
    """
    return sp.S(1)/sp.jacobi(n, alf, bet, 1)

def un(alf, bet, n):
    r"""Scaling function

    .. math::

        u_n^{(\alpha, \beta)} = \frac{n+1}{P^{(\alpha,\beta)}_n(1)}

    Parameters
    ----------
    alf, bet : numbers
        Jacobi parameters
    n : int
        Index

    Note
    ----
    Used by Chebyshev polynomials of second kind
    """
    return (n+1)/sp.jacobi(n, alf, bet, 1)

def qn(alf, bet, n, gn=cn):
    r"""Specialized Jacobi polynomial

    .. math::

        Q_n^{(\alpha, \beta)} = g_n^{(\alpha, \beta)} P^{(\alpha, \beta)}_n

    Parameters
    ----------
    alf, bet : numbers
        Jacobi parameters
    n : int
        Index
    gn : scaling function, optional
        Chebyshev of first and second kind use cn and un, respectively.
        Legendre uses gn=1.
    """
    if gn == 1:
        return jt(alf, bet, n)
    return gn(alf, bet, n)*jt(alf, bet, n)

def dqn(alf, bet, n, k, gn=cn):
    r"""Derivative of specialized Jacobi polynomial

    .. math::

        \frac{d^k Q_n^{(\alpha, \beta)}}{dx^k} = g_n^{(\alpha, \beta)} \frac{d^k P^{(\alpha, \beta)}_n}{dx^k}

    Parameters
    ----------
    alf, bet : numbers
        Jacobi parameters
    n : int
        Index
    k : int
        The k'th derivative
    gn : scaling function, optional
        Chebyshev of first and second kind use cn and un, respectively.
        Legendre uses gn=1.
    """
    if gn == 1:
        return djt(alf, bet, n, k)
    return gn(alf, bet, n)*djt(alf, bet, n, k)

def bnd_values(alf, bet, k=0, gn=1):
    r"""Return lambda function for computing boundary values

    .. math::

        \frac{d^k}{dx^k}Q_n(\pm 1)

    where :math:`Q^{(\alpha,\beta)}_n = g^{(\alpha,\beta)}_n P_n^{(\alpha,\beta)}`.

    Parameters
    ----------
    alf, bet : Numbers
        Jacobi parameters
    k : int, optional
        Number of derivatives
    gn : scaling function, optional
    """
    if gn == 1:
        gn = lambda a, b, n: 1

    if k == 0:
        return (lambda i: gn(alf, bet, i)*(-1)**i*sp.binomial(i+bet, i), lambda i: gn(alf, bet, i)*sp.binomial(i+alf, i))
    elif k == 1:
        gam = lambda i: sp.rf(i+alf+bet+1, 1)*sp.Rational(1, 2)
        return (lambda i: gn(alf, bet, i)*(-1)**(i-1)*gam(i)*sp.binomial(i+bet, i-1), lambda i: gn(alf, bet, i)*gam(i)*sp.binomial(i+alf, i-1))
    elif k == 2:
        gam = lambda i: sp.rf(i+alf+bet+1, 2)*sp.Rational(1, 4)
        return (lambda i: gn(alf, bet, i)*(-1)**i*gam(i)*sp.binomial(i+bet, i-2), lambda i: gn(alf, bet, i)*gam(i)*sp.binomial(i+alf, i-2))
    elif k == 3:
        gam = lambda i: sp.rf(i+alf+bet+1, 3)*sp.Rational(1, 8)
        return (lambda i: gn(alf, bet, i)*(-1)**i*gam(i)*sp.binomial(i+bet, i-3), lambda i: gn(alf, bet, i)*gam(i)*sp.binomial(i+alf, i-3))
    elif k == 4:
        gam = lambda i: sp.rf(i+alf+bet+1, 4)*sp.Rational(1, 16)
        return (lambda i: gn(alf, bet, i)*(-1)**i*gam(i)*sp.binomial(i+bet, i-4), lambda i: gn(alf, bet, i)*gam(i)*sp.binomial(i+alf, i-4))
    raise RuntimeError

def _a(alf, bet, i, j):
    """Matrix A for non-normalized Jacobi polynomials
    """
    return (
        sp.S(2)*(j+alf)*(j+bet)/((sp.S(2)*j+alf+bet+1)*(sp.S(2)*j+alf+bet))*delta(i+1, j)-
        (alf**2-bet**2)/((sp.S(2)*j+alf+bet+sp.S(2))*(sp.S(2)*j+alf+bet))*delta(i, j)+
        sp.S(2)*(j+1)*(j+alf+bet+sp.S(1))/((sp.S(2)*j+alf+bet+2)*(sp.S(2)*j+alf+bet+sp.S(1)))*delta(i-1, j)
    )

def _b(alf, bet, i, j):
    """Matrix B for non-normalized Jacobi polynomials"""
    f = ((sp.S(2)*(i+alf+bet) / ((sp.S(2)*i+alf+bet)*(sp.S(2)*i+alf+bet-sp.S(1)))) *delta(i, j+1)+
         -((sp.S(2)*(i+alf+sp.S(1))*(i+bet+sp.S(1))) / ((sp.S(2)*i+alf+bet+sp.S(3))*(sp.S(2)*i+alf+bet+sp.S(2))*(i+alf+bet+sp.S(1))))*delta(i, j-1))
    if alf != bet:
        f += ((sp.S(2)*(alf**2-bet**2)) / ((alf+bet)*(sp.S(2)*i+alf+bet+sp.S(2))*(sp.S(2)*i+alf+bet)))*delta(i, j)
    return f

def _c(alf, bet, i, j):
    """Matrix C for non-normalized Jacobi polynomials"""
    f = (sp.S(2)*(i+alf+1)*(i+bet+sp.S(1))*(i+alf+bet+sp.S(2))/((sp.S(2)*i+alf+bet+sp.S(2))*(sp.S(2)*i+alf+bet+sp.S(3)))*delta(i, j-1)
         -sp.S(2)*(i-sp.S(1))*i*(i+alf+bet)/((sp.S(2)*i+alf+bet-sp.S(1))*(sp.S(2)*i+alf+bet))*delta(i, j+1))
    if alf != bet:
        f += sp.S(2)*i*(alf-bet)*(i+alf+bet+sp.S(1))/((sp.S(2)*i+alf+bet)*(sp.S(2)*i+alf+bet+sp.S(2)))*delta(i, j)
    return f

def a(alf, bet, i, j, gn=1):
    r"""Recursion matrix :math:`A` for standardized Jacobi polynomials

    The recursion is

    .. math::

        x \boldsymbol{Q} = {A}^T \boldsymbol{Q}

    where

    .. math::

        Q_n(x) = g_n^{(\alpha,\beta)} P^{(\alpha,\beta)}_n(x) \\
        \boldsymbol{Q} = (Q_0, Q_1, \ldots, Q_{N})^T

    Parameters
    ----------
    alf, bet : numbers
        Jacobi parameters
    i, j : int
        Indices for row and column
    gn : scaling function, optional
        Chebyshev of first and second kind use cn and un, respectively.
        Legendre uses gn=1.
    """
    f = _a(alf, bet, i, j)
    return f if gn == 1 else gn(alf, bet, j) / gn(alf, bet, i) * f

def b(alf, bet, i, j, gn=1):
    r"""Recursion matrix :math:`B` for standardized Jacobi polynomials

    The recursion is

    .. math::

        \boldsymbol{Q} = {B}^T \partial \boldsymbol{Q}

    where :math:`\partial` represents the derivative and

    .. math::

        Q_n(x) = g_n^{(\alpha,\beta)} P^{(\alpha,\beta)}_n(x) \\
        \boldsymbol{Q} = (Q_0, Q_1, \ldots, Q_{N})^T

    Parameters
    ----------
    alf, bet : numbers
        Jacobi parameters
    i, j : int
        Indices for row and column
    gn : scaling function, optional
        Chebyshev of first and second kind use cn and un, respectively.
        Legendre uses gn=1.

    """
    f = _b(alf, bet, i, j)
    return f if gn == 1 else gn(alf, bet, j) / gn(alf, bet, i) * f

def c(alf, bet, i, j, gn=1):
    r"""Recursion matrix :math:`C` for standardized Jacobi polynomials

    The recursion is

    .. math::

        (1-x^2) \partial \boldsymbol{Q} = {C}^T \boldsymbol{Q}

    where :math:`\partial` represents the derivative and

    .. math::

        Q_n(x) = g_n^{(\alpha,\beta)} P^{(\alpha,\beta)}_n(x) \\
        \boldsymbol{Q} = (Q_0, Q_1, \ldots, Q_{N})^T

    Parameters
    ----------
    alf, bet : numbers
        Jacobi parameters
    i, j : int
        Indices for row and column
    gn : scaling function, optional
        Chebyshev of first and second kind use cn and un, respectively.
        Legendre uses gn=1.

    """
    f = _c(alf, bet, i, j)
    return f if gn == 1 else gn(alf, bet, j) / gn(alf, bet, i) * f

def psi(alf, bet, n, k):
    r"""Normalization factor for

    .. math::

        \partial^k P^{(\alpha, \beta)}_n = \psi^{(k,\alpha,\beta)}_{n} P^{(\alpha+k,\beta+k)}_{n-k}, \quad n \ge k, \quad (*)

    where :math:`\partial^k` represents the :math:`k`'th derivative

    Parameters
    ----------
    alf, bet : numbers
        Jacobi parameters
    n, k : int
        Parameters in (*)
    """
    return sp.rf(n+alf+bet+1, k) / 2**k

def a_(k, q, alf, bet, i, j, gn=1):
    r"""Recursion matrix :math:`\underline{A}` for standardized Jacobi polynomials

    The recursion is

    .. math::

        x \partial^k \boldsymbol{Q} = \underline{A}^T \partial^k \boldsymbol{Q} \quad (*)

    where :math:`\partial^k` represents the :math:`k`'th derivative and

    .. math::

        Q_n(x) = g_n^{(\alpha,\beta)} P^{(\alpha,\beta)}_n(x) \\
        \boldsymbol{Q} = (Q_0, Q_1, \ldots, Q_{N})^T

    Parameters
    ----------
    k, q : int
        Parameters in (*)
    alf, bet : numbers
        Jacobi parameters
    i, j : int
        Indices for row and column
    gn : scaling function, optional
        Chebyshev of first and second kind use cn and un, respectively.
        Legendre uses gn=1.
    """
    f = psi(alf, bet, j, k)/psi(alf, bet, i, k)*matpow(a, q, alf+k, bet+k, i-k, j-k)
    return f if gn == 1 else gn(alf, bet, j) / gn(alf, bet, i) * f

def gamma(alf, bet, n):
    r"""Return normalization factor :math:`h_n` for inner product of Jacobi polynomials

    .. math::

        h_n = (P^{(\alpha,\beta)}_n, P^{(\alpha,\beta)}_n)_{\omega^{(\alpha,\beta)}}

    Parameters
    ----------
    alf, bet : numbers
        Jacobi parameters
    n : int
        Index
    """
    #f = 2**(alfa+beta+1)*sp.gamma(m+alfa+1)*sp.gamma(m+beta+1)/sp.gamma(m+alfa+beta+1)/sp.gamma(m+1)/(2*m+alfa+beta+1)
    f = sp.rf(n+1, alfa)/sp.rf(n+beta+1, alfa) * 2**(alfa+beta+1)/(2*n+alfa+beta+1)
    return sp.simplify(f.subs([(alfa, alf), (beta, bet)]))

def h(alf, bet, n, k, gn=1):
    r"""Return normalization factor :math:`h^{(k)}_n` for inner product of derivatives of Jacobi polynomials

    .. math::

        Q_n(x) = g_n(x)P^{(\alpha,\beta)}_n(x) \\
        h_n^{(k)} = (\partial^k Q_n, \partial^k Q_n)_{\omega^{(\alpha+k,\beta+k)}} \quad (*)

    where :math:`\partial^k` represents the :math:`k`'th derivative.

    Parameters
    ----------
    alf, bet : numbers
        Jacobi parameters
    n : int
        Index
    k : int
        For derivative of k'th order, see (*)
    gn : scaling function, optional
        Chebyshev of first and second kind use cn and un, respectively.
        Legendre uses gn=1.
    """
    f = gamma(alf+k, bet+k, n-k)*(psi(alf, bet, n, k))**2
    return f if gn == 1 else sp.simplify(gn(alf, bet, n)**2*f)

def matpow(mat, q, alf, bet, i, j, gn=1):
    """Compute and return component of q'th matrix power of mat

    Parameters
    ----------
    mat : Python function (a, b or c)
    q : int
        matrix power
    alf, bet : numbers
        Jacobi parameters
    i, j : int
        Row and column indices
    gn : scaling function, optional
        Chebyshev of first and second kind use cn and un, respectively.
        Legendre uses gn=1.

    """
    assert q < 7
    m = _matpow(mat, mat, q, alf, bet, i, j)
    return m if gn == 1 else gn(alf, bet, j) / gn(alf, bet, i) * m

def _matpow(mat, m2, q, alf, bet, i, j):

    if q == 0:
        return delta(i, j)

    if q == 1:
        return mat(alf, bet, i, j)

    def d2(alf, bet, i, j):
        return (
            mat(alf, bet, i, i-1)*m2(alf, bet, i-1, j)+
            mat(alf, bet, i, i)*m2(alf, bet, i, j)+
            mat(alf, bet, i, i+1)*m2(alf, bet, i+1, j))

    if q == 2:
        return d2(alf, bet, i, j)

    return _matpow(mat, d2, q-1, alf, bet, i, j)

_pmat = {}

#@profile
def pmat(mat, q, alf, bet, M, N, gn=1):
    r"""Return SparseMatrix of q'th matrix power of recursion matrix mat

    Parameters
    ----------
    mat : Python function for matrix
        The Python function must have signature (alf, bet, i, j, gn=1)
    q : int
        Matrix power
    alf, bet : numbers
        Jacobi parameters
    M, N : int
        Shape of returned matrix
    gn : scaling function, optional
        Chebyshev of first and second kind use cn and un, respectively.
        Legendre uses gn=1.

    """
    try:
        d = _pmat[(mat, q, alf, bet, M, N, gn)]
        return SparseMatrix(deepcopy(d), (M, N))
    except:
        pass

    d = {}
    for i in range(-q, q+1):
        f = matpow(mat, q, alfa, beta, m, m+i, gn)
        fz = sp.simplify(f.subs([(alfa, alf), (beta, bet)]))
        if not fz == 0:
            fz = sp.lambdify(m, fz)
            if i >= 0:
                Z = min(N-i, M)
                d[i] = np.zeros(Z)
                if mat == b:
                    d[i][:q] = 0
                    if len(d[i]) > q:
                        d[i][q] = f.subs(m, q).subs([(alfa, alf), (beta, bet)])
                    if len(d[i]) > q+1:
                        d[i][q+1:] = fz(np.arange(q+1, Z))
                else:
                    d[i][:q] = [f.subs(m, z).subs([(alfa, alf), (beta, bet)]) for z in np.arange(0, q)]
                    d[i][q:] = fz(np.arange(q, Z))
            else:
                Z = min(M-abs(i), N)
                d[i] = np.zeros(Z)
                if mat == b and q+i > 0:
                    d[i][:(q+i)] = 0
                    if alf == -half and bet == -half:
                        d[i][(q+i):q] = [sp.simplify(f.subs(m, -i+z)).subs([(alfa, alf), (beta, bet)]) for z in np.arange(q+i, q)]
                        d[i][q:] = fz(np.arange(-i+q, Z-i))
                    else:
                        d[i][(q+i):] = fz(np.arange(q, Z-i))

                else:
                    if alf == -half and bet == -half:
                        d[i][:q] = [sp.simplify(f.subs(m, -i+z)).subs([(alfa, alf), (beta, bet)]) for z in np.arange(0, q)]
                        d[i][q:] = fz(np.arange(-i+q, Z-i))
                    else:
                        d[i][:] = fz(np.arange(-i, Z-i))
    _pmat[(mat, q, alf, bet, M, N, gn)] = d
    return SparseMatrix(d, (M, N))

_amat = {}

def a_mat(mat, k, q, alf, bet, M, N, gn=1):
    r"""Return SparseMatrix of recursion matrix

    .. math::

        A^{(k,q)} = (a^{(k,q)}_{mn})_{m,n=0}^{M, N}

    Parameters
    ----------
    mat : Python function for matrix
        The Python function must have signature (k, q, alf, bet, i, j, gn=1)
    k : int
        Parameter
    q : int
        Matrix power
    alf, bet : numbers
        Jacobi parameters
    M, N : int
        Shape of returned matrix
    gn : scaling function, optional
        Chebyshev of first and second kind use cn and un, respectively.
        Legendre uses gn=1.

    """
    try:
        d = _amat[(mat, k, q, alf, bet, M, N, gn)]
        return SparseMatrix(deepcopy(d), (M, N))
    except:
        pass
    d = {}
    for i in range(-q, q+1):
        f = mat(k, q, alfa, beta, m, m+i, gn)
        fz = sp.simplify(f.subs(alfa, alf).subs(beta, bet))
        if not fz == 0:
            fz = sp.lambdify(m, fz)
            if i >= 0:
                Z = min(N-abs(i), M)
                if Z > 0:
                    d[i] = np.zeros(Z)
                    if len(d[i]) > k:
                        d[i][k:k+q] = [f.subs(m, z).subs([(alfa, alf), (beta, bet)]) for z in np.arange(k, min(k+q, len(d[i])))]
                    if len(d[i]) > k+q:
                        d[i][k+q:] = fz(np.arange(k+q, Z))
            else:
                Z = min(M-abs(i), N)
                if Z > 0:
                    d[i] = np.zeros(Z)
                    d[i][k] = sp.simplify(f.subs(m, -i+k)).subs([(alfa, alf), (beta, bet)])
                    d[i][k+1:] = fz(np.arange(-i+k+1, Z-i))
    _amat[(mat, k, q, alf, bet, M, N, gn)] = d
    return SparseMatrix(d, (M, N))

def ShiftedMatrix(mat, q, r, s, M=0, N=0, k=None, alf=0, bet=0, gn=1):
    r"""Index-shifted q'th power of matrix `mat`. Either

    .. math::

        A^{(k, q)}_{(r, s)} = (a^{(k, q)}_{m+r,n+s})_{m,n=0}^{M,N} \\
        B^{(q)}_{(r, s)} = (b^{(q)}_{m+r,n+s})_{m,n=0}^{M,N}

    Parameters
    ----------
    mat : Python function or SparseMatrix
        The Python function must have signature (k, q, alf, bet, i, j, gn=1)
        or (alf, bet, i, j, gn=1) if `k` is None
    q : int
        The matrix power
    r : int
        Shift in row
    s : int
        Shift in column
    M, N : int, optional
        The shape of the final matrix
    k : int or None, optional
        Parameter of recursion of the k'th derivative
        This is only used if the recursion matrix is A.
    alf, bet : numbers, optional
        Jacobi paramters
    gn : scaling function, optional
        Chebyshev of first and second kind use cn and un, respectively.
        Legendre uses gn=1.

    Note
    ----
    If `mat` is a SparseMatrix, then the resulting shifted matrix will
    be smaller than `mat`.
    """
    if isinstance(mat, SparseMatrix):
        A = mat
        M, N = A.shape
        M, N = M-r, N-s
    elif k is not None:
        A = a_mat(mat, k, q, alf, bet, M+max(r, s), N+max(r, s), gn)
    else:
        A = pmat(mat, q, alf, bet, M+max(r, s), N+max(r, s), gn)
    d = {}
    for key, val in A.items():
        nkey = key+r-s
        j = max(r+min(key, 0), s-max(key, 0))
        if nkey >= 0:
            d[nkey] = val[j:min(N-nkey, M)+j].copy()
        else:
            d[nkey] = val[j:min(M+nkey, N)+j].copy()
        if len(d[nkey]) == 0:
            del d[nkey]
    return SparseMatrix(d, (M, N))

def Lmat(k, q, l, M, N, alf=0, bet=0, gn=1):
    r"""
    Return matrix corresponding to

    .. math::

        (\partial^{k-l}Q_{n}, x^q \phi^{(k)}_m)_{\omega}\quad (1)

    where :math:`\partial^k` represents the :math:`k`'th derivative and

    .. math::

        Q_n = g_n^{(\alpha, \beta)} P^{(\alpha, \beta)}_n \\
        \phi^{(k)}_m = \frac{(1-x^2)^k \partial^k Q_{m+k}}{h^{(k)}_{m+k}}

    Parameters
    ----------
    k, q, l : integers
        Numbers in variational form (1)
    M, N : int
        Shape of matrix
    alf, bet : numbers, optional
        Jacobian parameters
    gn : scaling function
        Chebyshev of first and second kind use cn and un, respectively.
        Legendre uses gn=1.

    Example
    -------
    >>> from shenfun.jacobi.recursions import Lmat, cn, half
    >>> X = Lmat(2, 2, 0, 10, 12, -half, -half, cn)

    """

    if l == 0 and q == 0:
        return SparseMatrix({k: 1}, (M, N)).diags('csr')
    elif l == 0:
        A = ShiftedMatrix(a_, q, k, k, M, N, k=k, alf=alf, bet=bet, gn=gn).diags('csr')
        return A*SparseMatrix({k: 1}, (N, N)).diags('csr')
    elif q == 0:
        return ShiftedMatrix(b, l, k, 0, M, N, alf=alf, bet=bet, gn=gn).diags('csr')
    else:
        A = ShiftedMatrix(a_, q, k, k, M, N, k=k, alf=alf, bet=bet, gn=gn).diags('csr')
        B = ShiftedMatrix(b, l, k, 0, N, N, alf=alf, bet=bet, gn=gn).diags('csr')
        return A*B
