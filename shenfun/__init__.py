import numpy as np
from . import chebyshev
from . import legendre
from . import matrixbase
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve as lasolve

def inner_product(test, trial, N):
    """Return inner product of bilinear form

    args:
        test     (Basis, integer)     Basis is any of the classes from
                                      shenfun.chebyshev.bases or
                                      shenfun.legendre.bases
                                      The integer determines the numer of times
                                      the basis is differentiated.
                                      The test represents the matrix row
        trial    (Basis, integer)     As test, but representing matrix column

    Example:
        Compute mass matrix of Shen's Chebyshev Dirichlet basis:

        >>> from shenfun.chebyshev.bases import ShenDirichletBasis
        >>> SD = ShenDirichletBasis()
        >>> B = inner_product((SD, 0), (SD, 0), 6)
        >>> B
        {-2: array([-1.57079633]),
          0: array([ 4.71238898,  3.14159265,  3.14159265,  3.14159265]),
          2: array([-1.57079633])}

    """
    assert trial[0].__module__ == test[0].__module__
    k = np.arange(N).astype(np.float)
    if isinstance(test[0], chebyshev.ChebyshevBase):
        return chebyshev.mat[(test, trial)](k)

    elif isinstance(test[0], legendre.LegendreBase):
        return legendre.mat[(test, trial)](k)

def solve(A, b, u=None, axis=0):
    """Solve Au=b and return u

    The matrix A must be square

    args:
        A                      SparseMatrix
        u   (output)           Array
        b   (input/output)     Array

    """
    assert A.shape[0] == A.shape[1]
    assert isinstance(A, matrixbase.SparseMatrix)
    s = A.testfunction[0].slice(b.shape[axis])

    uc = False
    if u is None:
        uc = True
        u = np.zeros_like(b)
    else:
        assert u.shape == b.shape

    # Move axis to first
    if axis > 0:
        b = np.moveaxis(b, axis, 0)
        u = np.moveaxis(u, axis, 0)

    bs = b[s]
    us = u[s]
    assert A.shape[0] == bs.shape[0]
    if isinstance(A.testfunction[0], chebyshev.bases.ShenNeumannBasis):
        # Handle level by using Dirichlet for dof=0
        Aa = A.diags().toarray()
        Aa[0] = 0
        Aa[0,0] = 1
        b[0] = A.testfunction[0].mean
        if b.ndim == 1:
            us[:] = lasolve(Aa, bs)
        else:
            N = bs.shape[0]
            P = np.prod(bs.shape[1:])
            us[:] = lasolve(Aa, bs.reshape((N, P))).reshape(bs.shape)

    else:
        if b.ndim == 1:
            u[s] = spsolve(A.diags('csr'), b[s])
        else:
            N = bs.shape[0]
            P = np.prod(bs.shape[1:])
            br = bs.reshape((N, P))

            if b.dtype is np.dtype('complex'):
                us.real[:] = spsolve(A.diags('csr'), br.real).reshape(bs.shape)
                us.imag[:] = spsolve(A.diags('csr'), br.imag).reshape(bs.shape)
            else:
                us[:] = spsolve(A.diags('csr'), br).reshape(us.shape)

    if uc is True:
        b[:] = u
        if axis > 0:
            b = np.moveaxis(b, 0, axis)
        return b

    else:
        if axis > 0:
            u = np.moveaxis(u, 0, axis)
        return u
