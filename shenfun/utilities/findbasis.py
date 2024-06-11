import importlib
import functools
from sympy.matrices.common import NonInvertibleMatrixError
import sympy as sp
import numpy as np

n = sp.Symbol('n', real=True, integer=True, positive=True)

def get_stencil_matrix(bcs, family, alpha=None, beta=None, gn=1):
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

        - ``Chebyshev``
        - ``Chebyshevu``
        - ``Legendre``
        - ``Ultraspherical``
        - ``Jacobi``
        - ``Laguerre``

    alpha, beta : numbers, optional
        The Jacobi parameters, used only for ``Jacobi`` or ``Ultraspherical``

    gn : Scaling function for Jacobi polynomials

    """
    from shenfun.spectralbase import BoundaryConditions
    base = importlib.import_module('.'.join(('shenfun', family.lower())))
    bnd_values = functools.partial(base.Orthogonal.bnd_values, alpha=alpha, beta=beta, gn=gn)
    bcs = BoundaryConditions(bcs)
    bc = {'D': 0, 'N': 1, 'N2': 2, 'N3': 3, 'N4': 4}
    lr = {'L': 0, 'R': 1}
    lra = {'L': 'left', 'R': 'right'}
    s = []
    r = []
    for key in bcs.orderednames():
        k, v = key[0], key[1:]
        if v in 'WR':
            k0 = 0 if v == 'R' else 1
            alfa = bcs[lra[k]][v][0]
            f = [bnd_values(k=k0)[lr[k]], bnd_values(k=k0+1)[lr[k]]]
            s.append([sp.simplify(f[0](n+j)+alfa*f[1](n+j)) for j in range(1, 1+bcs.num_bcs())])
            r.append(-sp.simplify(f[0](n)+alfa*f[1](n)))
        else:
            f = bnd_values(k=bc[v])[lr[k]]
            s.append([sp.simplify(f(n+j)) for j in range(1, 1+bcs.num_bcs())])
            r.append(-sp.simplify(f(n)))
    A = sp.Matrix(s)
    b = sp.Matrix(r)
    M = sp.simplify(A.solve(b))
    d = {0: 1}
    for i, s in enumerate(M):
        if not s == 0:
            d[i+1] = s
    return d

def get_bc_basis(bcs, family, alpha=None, beta=None, gn=1):
    """Return boundary basis satisfying `bcs`.

    Parameters
    ----------
    bcs : dict
        The boundary conditions in dictionary form, see
        :class:`.BoundaryConditions`.

    family : str
        Choose one of

        - ``Chebyshev``
        - ``Chebyshevu``
        - ``Legendre``
        - ``Ultraspherical``
        - ``Jacobi``
        - ``Laguerre``

    alpha, beta : numbers, optional
        The Jacobi parameters, used only for ``Jacobi`` or ``Ultraspherical``

    gn : Scaling function for Jacobi polynomials

    """
    from shenfun.spectralbase import BoundaryConditions
    base = importlib.import_module('.'.join(('shenfun', family.lower())))
    bnd_values = functools.partial(base.Orthogonal.bnd_values, alpha=alpha, beta=beta, gn=gn)
    bcs = BoundaryConditions(bcs)
    def _computematrix(first):
        bc = {'D': 0, 'N': 1, 'N2': 2, 'N3': 3, 'N4': 4}
        lr = {'L': 0, 'R': 1}
        lra = {'L': 'left', 'R': 'right'}
        s = []
        for key in bcs.orderednames():
            k, v = key[0], key[1:]
            if v in 'WR':
                k0 = 0 if v == 'R' else 1
                alfa = bcs[lra[k]][v][0]
                f = [bnd_values(k=k0)[lr[k]], bnd_values(k=k0+1)[lr[k]]]
                s.append([sp.simplify(f[0](j)+alfa*f[1](j)) for j in range(first, first+bcs.num_bcs())])
            else:
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
            continue

    sol = sp.Matrix(np.zeros((bcs.num_bcs(), first+bcs.num_bcs())))
    sol[:, first:] = s
    return sol
