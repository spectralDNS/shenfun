import numpy as np
from . import chebyshev
from . import legendre
from . import fourier
from . import matrixbase
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve as lasolve

def inner_product(test, trial, out=None, axis=0, fast_transform=False):
    """Return inner product of linear or bilinear form

    args:
        test     (Basis, integer)     Basis is any of the classes from
                                      shenfun.chebyshev.bases,
                                      shenfun.legendre.bases or
                                      shenfun.fourier.bases
                                      The integer determines the numer of times
                                      the basis is differentiated.
                                      The test represents the matrix row
        trial    (Basis, integer)     As test, but representing matrix column
                       or
                    function          Function evaluated at quadrature nodes
                                      (for linear forms)

    kwargs:
        out          Numpy array      Return array
        axis             int          Axis to take the inner product over

    Example:
        Compute mass matrix of Shen's Chebyshev Dirichlet basis:

        >>> from shenfun.chebyshev.bases import ShenDirichletBasis
        >>> SD = ShenDirichletBasis(6)
        >>> B = inner_product((SD, 0), (SD, 0))
        >>> B
        {-2: array([-1.57079633]),
          0: array([ 4.71238898,  3.14159265,  3.14159265,  3.14159265]),
          2: array([-1.57079633])}

    """
    if isinstance(test, tuple):
        # Bilinear form
        assert trial[0].__module__ == test[0].__module__
        key = ((test[0].__class__, test[1]), (trial[0].__class__, trial[1]))
        if isinstance(test[0], chebyshev.ChebyshevBase):
            return chebyshev.mat[key](test, trial)

        elif isinstance(test[0], legendre.LegendreBase):
            return legendre.mat[key](test, trial)

        elif isinstance(test[0], fourier.FourierBase):
            return fourier.mat[key](test, trial)

    else:
        # Linear form
        if out is None:
            sl = list(trial.shape)
            if isinstance(test, fourier.FourierBase):
                if isinstance(test, fourier.R2CBasis):
                    sl[axis] = sl[axis]//2+1
                out = np.zeros(sl, dtype=np.complex)
            else:
                out = np.zeros_like(trial)
        out = test.scalar_product(trial, out, axis=axis, fast_transform=fast_transform)
        return out

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
            u[-1] = A.testfunction[0].bc[0]
            u[-2] = A.testfunction[0].bc[1]

    if axis > 0:
        u = np.moveaxis(u, 0, axis)
        b = np.moveaxis(b, 0, axis)

    return u
