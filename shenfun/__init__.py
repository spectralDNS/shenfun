import numpy as np
from . import chebyshev
from . import legendre
from . import fourier
from . import matrixbase
from .forms.inner import *
from .forms.operators import *
from .forms.arguments import *
from .tensorproductspace import *

from scipy.sparse.linalg import spsolve
from scipy.linalg import solve as lasolve


def solve(A, b, u=None, axis=0):
    """Solve Au=b and return u

    The matrix A must be square

    args:
        A                         SparseMatrix
        b      (input/output)     Array

    kwargs:
        u      (output)           Array
        axis       int            The axis to solve along

    If u is not provided, then b is overwritten with the solution and returned

    """
    assert A.shape[0] == A.shape[1]
    assert isinstance(A, matrixbase.SparseMatrix)
    s = A.testfunction[0].slice()

    if u is None:
        u = b
    else:
        assert u.shape == b.shape

    # Move axis to first
    if axis > 0:
        u = np.moveaxis(u, axis, 0)
        if not u is b:
            b = np.moveaxis(b, axis, 0)

    assert A.shape[0] == b[s].shape[0]
    if (isinstance(A.testfunction[0], chebyshev.bases.ShenNeumannBasis) or
        isinstance(A.testfunction[0], legendre.bases.ShenNeumannBasis)):
        # Handle level by using Dirichlet for dof=0
        Aa = A.diags().toarray()
        Aa[0] = 0
        Aa[0,0] = 1
        b[0] = A.testfunction[0].mean
        if b.ndim == 1:
            u[s] = lasolve(Aa, b[s])
        else:
            N = b[s].shape[0]
            P = np.prod(b[s].shape[1:])
            u[s] = lasolve(Aa, b[s].reshape((N, P))).reshape(b[s].shape)

    else:
        if b.ndim == 1:
            u[s] = spsolve(A.diags('csr'), b[s])
        else:
            N = b[s].shape[0]
            P = np.prod(b[s].shape[1:])
            br = b[s].reshape((N, P))

            if b.dtype is np.dtype('complex'):
                u.real[s] = spsolve(A.diags('csr'), br.real).reshape(u[s].shape)
                u.imag[s] = spsolve(A.diags('csr'), br.imag).reshape(u[s].shape)
            else:
                u[s] = spsolve(A.diags('csr'), br).reshape(u[s].shape)
        if hasattr(A.testfunction[0], 'bc'):
            A.testfunction[0].bc.apply_after(u, True)

    if axis > 0:
        u = np.moveaxis(u, 0, axis)
        if not u is b:
            b = np.moveaxis(b, 0, axis)

    u /= A.scale

    return u
