"""
Recursions for Jacobi polynomials, or standardized Jacobi polynomials

"""
from shenfun.matrixbase import SparseMatrix
import numpy as np
import sympy as sp

x = sp.Symbol('x', real=True)
delta = sp.KroneckerDelta
half = sp.Rational(1, 2)
m, n, k = sp.symbols('m,n,k', real=True, integer=True)
alfa, beta = sp.symbols('a,b', real=True)

# Jacobi Jn
jt = lambda alf, bet, n: sp.jacobi(n, alf, bet, x)
djt = lambda alf, bet, n, k: sp.diff(jt(alf, bet, n), x, k)

# Scaled Qn = gn*Jn
cn = lambda alf, bet, n: sp.S(1)/sp.jacobi(n, alf, bet, 1)
un = lambda alf, bet, n: (n+1)/sp.jacobi(n, alf, bet, 1)
def qn(alf, bet, n, gn=cn):
    return gn(alf, bet, n)*jt(alf, bet, n)
def dqn(alf, bet, n, k, gn=cn):
    return gn(alf, bet, n)*djt(alf, bet, n, k)

def _a(alf, bet, i, j):
    """Matrix A for non-normalized Jacobi polynomials"""
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
    """Matrix A for standardized Jacobi polynomials"""
    f = _a(alf, bet, i, j)
    return f if gn == 1 else gn(alf, bet, j) / gn(alf, bet, i) * f

def b(alf, bet, i, j, gn=1):
    """Matrix B for standardized Jacobi polynomials"""
    f = _b(alf, bet, i, j)
    return f if gn == 1 else gn(alf, bet, j) / gn(alf, bet, i) * f

def c(alf, bet, i, j, gn=1):
    """Matrix C for standardized Jacobi polynomials"""
    f = _c(alf, bet, i, j)
    return f if gn == 1 else gn(alf, bet, j) / gn(alf, bet, i) * f

def psi(alf, bet, n, k):
    return sp.rf(n+alf+bet+1, k) / 2**k

def a_(k, q, alf, bet, i, j, gn=1):
    f = psi(alf, bet, j, k)/psi(alf, bet, i, k)*matpow(a, q, alf+k, bet+k, i-k, j-k)
    return f if gn == 1 else gn(alf, bet, j) / gn(alf, bet, i) * f

def gamma(alf, bet, n):
    f = 2**(alfa+beta+1)*sp.gamma(m+alfa+1)*sp.gamma(m+beta+1)/sp.gamma(m+alfa+beta+1)/sp.gamma(m+1)/(2*m+alfa+beta+1)
    return sp.simplify(f.subs(m, n)).subs([(alfa, alf), (beta, bet)])

def h(alf, bet, n, k, gn=1):
    """Return normalization factor"""
    f = gamma(alf+k, bet+k, n-k)*(psi(alf, bet, n, k))**2
    return sp.simplify(f) if gn == 1 else sp.simplify(gn(alf, bet, n)**2*f)

def matpow(mat, q, alf, bet, i, j, gn=1):
    """Compute and return q'th matrix power of mat"""
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


def pmat(mat, q, alf, bet, M, N, gn=1):
    """Return SparseMatrix of q'th matrix power of mat"""
    d = {}
    for i in range(-q, q+1):
        f = matpow(mat, q, alfa, beta, m, m+i, gn)
        fz = sp.simplify(f.subs([(alfa, alf), (beta, bet)]))
        if not fz == 0:
            fz = sp.lambdify(m, fz)
            if i >= 0:
                Z = min(N-abs(i), M)
                d[i] = np.zeros(Z)
                if mat == b:
                    d[i][:q] = 0
                    d[i][q] = f.subs(m, q).subs([(alfa, alf), (beta, bet)])
                    d[i][q+1:] = fz(np.arange(q+1, Z))
                else:
                    d[i][:q] = [f.subs(m, z).subs([(alfa, alf), (beta, bet)]) for z in np.arange(0, q)]
                    d[i][q:] = fz(np.arange(q, Z))
            else:
                Z = min(M-abs(i), N)
                d[i] = np.zeros(Z)
                if mat == b and q > 2:
                    d[i][:(q+i)] = 0
                    d[i][(q+i):q] = [sp.simplify(f.subs(m, -i+z)).subs([(alfa, alf), (beta, bet)]) for z in np.arange(q+i, q)]
                    d[i][q:] = fz(np.arange(-i+q, Z-i))
                else:
                    d[i][:q] = [sp.simplify(f.subs(m, -i+z)).subs([(alfa, alf), (beta, bet)]) for z in np.arange(0, q)]
                    d[i][q:] = fz(np.arange(-i+q, Z-i))

    return SparseMatrix(d, (M, N))

def a_mat(mat, k, q, alf, bet, M, N, gn=1):
    r"""Return SparseMatrix of

    .. math::

        A^{(k,q)} = (a^{(k,q)}_{mn})_{m,n=0}^{M, N}

    Parameters
    ----------
    mat : Python function for matrix
        The Python function must have signature (k, q, alf, bet, i, j, gn=1)

    """
    d = {}
    for i in range(-q, q+1):
        f = mat(k, q, alfa, beta, m, m+i, gn)
        fz = sp.simplify(f.subs(alfa, alf).subs(beta, bet))
        if not fz == 0:
            fz = sp.lambdify(m, fz)
            if i >= 0:
                Z = min(N-abs(i), M)
                d[i] = np.zeros(Z)
                d[i][k:k+q] = [f.subs(m, z).subs([(alfa, alf), (beta, bet)]) for z in np.arange(k, k+q)]
                d[i][k+q:] = fz(np.arange(k+q, Z))
            else:
                Z = min(M-abs(i), N)
                d[i] = np.zeros(Z)
                d[i][k] = sp.simplify(f.subs(m, -i+k)).subs([(alfa, alf), (beta, bet)])
                d[i][k+1:] = fz(np.arange(-i+k+1, Z-i))

    return SparseMatrix(d, (M, N))

def ShiftedMatrix(mat, q, r, s, M=0, N=0, k=None, alf=0, bet=0, gn=1):
    """Index-shifted q'th power of matrix `mat`

    .. math::

        A^{(k, q)}_{(r, s)} = (a^{(k, q)}_{m+r,n+s})_{m,n=0}^{M,N}

    Parameters
    ----------
    mat : Python function or SparseMatrix
        The Python function must have signature (k, q, alf, bet, i, j, gn=1)

    Note
    ----
    If `mat` is a SparseMatrix, then the resulting shifted matrix will
    be smaller than `mat`.
    """
    if isinstance(mat, SparseMatrix):
        A = mat
        M, N = A.shape
        M, N= M-r, N-s
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
    return SparseMatrix(d, (M, N))

def Lmat(k, q, l, M, N, alf=0, bet=0, gn=1):
    r"""
    Return matrix corrsponding to

    .. math::

        (\partial^{k-l}Q_{n}, x^q \phi^{(k)}_m)_{\omega} (1)

    where

    .. math::

        Q^{(\alpha, \beta)}_n = g_n^{(\alpha, \beta)} P^{(\alpha, \beta)}_n \\
        \phi^{(k)}_m = \frac{(1-x^2)^k \partial^k Q_{m+k}}{h^{(k)}_{m+k}}

    Parameters
    ----------
    k, q, l : integers
        Numbers in variational form (1)
    alf, bet : Jacobian parameters
    N : integer
        Number of quadrature points
    gn : scaling function

    Example
    -------
    >>> from shenfun.jacobi.recursions import Lmat, cn, half
    >>> X = Lmat(-half, -half, 2, 2, 0, 10, 12, cn)

    """

    if l == 0 and q == 0:
        return SparseMatrix({k:1}, (M, N)).diags('csr')
    elif l == 0:
        A = ShiftedMatrix(a_, q, k, k, M, N, k=k, alf=alf, bet=bet, gn=gn).diags('csr')
        return A*SparseMatrix({k: 1}, (N, N)).diags('csr')
    elif q == 0:
        return ShiftedMatrix(b, l, k, 0, M, N, alf=alf, bet=bet, gn=gn).diags('csr')
    else:
        A = ShiftedMatrix(a_, q, k, k, M, N, k=k, alf=alf, bet=bet, gn=gn).diags('csr')
        B = ShiftedMatrix(b, l, k, 0, N, N, alf=alf, bet=bet, gn=gn).diags('csr')
        return A*B
