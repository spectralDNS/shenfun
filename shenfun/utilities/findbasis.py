import importlib
import functools
from sympy.matrices.common import NonInvertibleMatrixError
import sympy as sp
import numpy as np

n = sp.Symbol('n', integer=True, positive=True)

def get_stencil_matrix(bcs, family, alpha=None, beta=None):
    """Return stencil matrix.

    Return the narrowest possible stencil matrix satisfying the
    homogeneous boundary conditions in `bcs`, and with leading
    coefficient equal to one.

    Parameters
    ----------
    bcs : dict
        The boundary conditions in dictionary form, see
        :class:`.BoundaryConditions`.

    family : str
        Choose one of

        - ``Chebyshev`` or ``C``
        - ``Chebyshevu`` or ``U``
        - ``Legendre`` or ``L``
        - ``Ultraspherical`` or ``Q``
        - ``Jacobi`` or ``J``
        - ``Laguerre`` or ``La``

    alpha, beta : numbers, optional
        The Jacobi parameters, used only for ``Jacobi`` or ``Ultraspherical``
    """
    from shenfun.spectralbase import BoundaryConditions
    base = importlib.import_module('.'.join(('shenfun', family.lower())))
    bnd_values = functools.partial(base.Orthogonal.bnd_values, alpha=alpha, beta=beta)
    bcs = BoundaryConditions(bcs)
    bc = {'D': 0, 'N': 1, 'N2': 2, 'N3': 3, 'N4': 4}
    lr = {'L': 0, 'R': 1}
    s = []
    r = []
    for key in bcs.orderednames():
        k, v = key[0], key[1:]
        f = bnd_values(k=bc[v])[lr[k]]
        s.append([sp.simplify(f(n+j)) for j in range(1, 1+bcs.num_bcs())])
        r.append(-sp.simplify(f(n)))
    A = sp.Matrix(s)
    b = sp.Matrix(r)
    return sp.simplify(A.solve(b))

def get_bc_basis(bcs, family, alpha=None, beta=None):
    """Return boundary basis satisfying `bcs`.

    Parameters
    ----------
    bcs : dict
        The boundary conditions in dictionary form, see
        :class:`.BoundaryConditions`.

    family : str
        Choose one of

        - ``Chebyshev`` or ``C``
        - ``Chebyshevu`` or ``U``
        - ``Legendre`` or ``L``
        - ``Ultraspherical`` or ``Q``
        - ``Jacobi`` or ``J``
        - ``Laguerre`` or ``La``

    alpha, beta : numbers, optional
        The Jacobi parameters, used only for ``Jacobi`` or ``Ultraspherical``
    """
    from shenfun.spectralbase import BoundaryConditions
    base = importlib.import_module('.'.join(('shenfun', family.lower())))
    bnd_values = functools.partial(base.Orthogonal.bnd_values, alpha=alpha, beta=beta)
    bcs = BoundaryConditions(bcs)
    def _computematrix(first):
        bc = {'D': 0, 'N': 1, 'N2': 2, 'N3': 3, 'N4': 4}
        lr = {'L': 0, 'R': 1}
        s = []
        for key in bcs.orderednames():
            k, v = key[0], key[1:]
            f = bnd_values(k=bc[v])[lr[k]]
            s.append([sp.simplify(f(j)) for j in range(first, first+bcs.num_bcs())])

        A = sp.Matrix(s)
        s = sp.simplify(A.solve(sp.eye(bcs.num_bcs())).T)
        return s

    first_basis = bcs.num_derivatives() // bcs.num_bcs()
    first = 0
    for first in range(first_basis+1):
        try:
            s = _computematrix(first)
            break
        except NonInvertibleMatrixError:
            #print('Not invertible using basis %d'%(first))
            continue

    sol = sp.Matrix(np.zeros((bcs.num_bcs(), first+bcs.num_bcs())))
    sol[:, first:] = s
    return sol
