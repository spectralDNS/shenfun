"""
This script has been used to compute the Dirichlet results of the paper

    On efficient Chebyshev-Galerkin methods for second-order equations

The results have been computed using Python 3.9 and Shenfun 3.2.2.
The results have been computed with the Numba optimization you get
by calling this script with argument --numba, e.g.,

   python CGpaper_naumann.py --return_type 2 --numba

The generalized Chebyshev-Tau results are computed with dedalus,
and are as such not part of this script.

"""
import sympy as sp
import numpy as np
import scipy.sparse.linalg as lin
import array_to_latex as a2l
from time import time

x = sp.Symbol('x', real=True)

fe = {}
rnd = {}
func = {}

def matvec(u_hat, f_hat, A, B, alpha, method):
    """Compute matrix vector product

    Parameters
    ----------
    u_hat : Function
        The solution array
    f_hat : Function
        The right hand side array
    A : SparseMatrix
        The stiffness matrix
    B : SparseMatrix
        The mass matrix
    alpha : number
        The weight of the mass matrix
    method : int
        The chosen method

    """
    from shenfun import chebyshev, la
    if method == 1:
        if alpha == 0:
            A.scale *= -1
            f_hat = A.matvec(u_hat, f_hat)
            A.scale *= -1
        else:
            sol = chebyshev.la.Helmholtz(A, B, -1, alpha)
            f_hat = sol.matvec(u_hat, f_hat)
    else:
        if alpha == 0:
            A.scale *= -1
            f_hat = A.matvec(u_hat, f_hat)
            A.scale *= -1
        else:
            M = alpha*B - A
            f_hat = M.matvec(u_hat, f_hat)
    return f_hat

def get_solver(A, B, alpha, method):
    """Return optimal solver for given method

    Parameters
    ----------
    A : SparseMatrix
        The stiffness matrix
    B : SparseMatrix
        The mass matrix
    alpha : number
        The weight of the mass matrix
    method : int
        The chosen method
    """
    from shenfun import chebyshev, la
    if method == 2:
        if alpha == 0:
            sol = la.TDMA(A*(-1))
        else:
            sol = la.PDMA(alpha*B - A)
    elif method == 1:
        if alpha == 0:
            A.scale = -1
            sol = chebyshev.la.ADDSolver(A)
        else:
            sol = chebyshev.la.Helmholtz(A, B, -1, alpha)
    elif method in (0, 3, 4):
        if alpha == 0:
            sol = la.TwoDMA(A*(-1))
        else:
            sol = la.FDMA(alpha*B-A)
    elif method == 5:
        if alpha == 0:
            AA = A*(-1)
            sol = AA.solve
        else:
            sol = la.TDMA(alpha*B-A)

    else:
        raise NotImplementedError
    return sol

def solve(f_hat, u_hat, A, B, alpha, method):
    """Solve (alpha*B-A)u_hat = f_hat

    Parameters
    ----------
    f_hat : Function
        The right hand side array
    u_hat : Function
        The solution array
    A : SparseMatrix
        The stiffness matrix
    B : SparseMatrix
        The mass matrix
    alpha : number
        The weight of the mass matrix
    method : int
        The chosen method

    """
    from shenfun import extract_bc_matrices, Function
    if isinstance(B, list):
        u_hat.set_boundary_dofs()
        bc_mat = extract_bc_matrices([B])
        B = B[0]
        w0 = Function(u_hat.function_space())
        f_hat -= alpha*bc_mat[0].matvec(u_hat, w0)

    sol = get_solver(A, B, alpha, method)
    u_hat = sol(f_hat, u_hat)
    return u_hat

def main(N, method=0, alpha=0, returntype=0):
    from shenfun import FunctionSpace, TrialFunction, TestFunction, \
        inner, div, grad, chebyshev, SparseMatrix, Function, Array
    global fe
    basis = {0: ('ShenDirichlet', 'Heinrichs'),
             1: ('ShenDirichlet', 'ShenDirichlet'),
             2: ('Heinrichs', 'Heinrichs'),
             3: ('DirichletU', 'ShenDirichlet'),
             4: ('Orthogonal', 'ShenDirichlet'),    # Quasi-Galerkin
             5: ('ShenDirichlet', 'ShenDirichlet'), # Legendre
             }

    test, trial = basis[method]
    if returntype == 2:
        ue = sp.sin(100*sp.pi*x)

    family = 'C' if method < 5 else 'L'
    kw = {}
    scaled = True if method in (0, 5) else False
    if scaled:
        kw['scaled'] = True
    ST = FunctionSpace(N, family, basis=test, **kw)
    TS = FunctionSpace(N, family, basis=trial, **kw)
    wt = {0: 1, 1: 1, 2: 1, 3: 1-x**2, 4: 1, 5: 1}[method]
    u = TrialFunction(TS)
    v = TestFunction(ST)
    A = inner(v*wt, div(grad(u)))
    B = inner(v*wt, u)

    if method == 4:
        # Quasi
        Q2 = chebyshev.quasi.QIGmat(N)
        A = Q2*A
        B = Q2*B
    if method == 3:
        k = np.arange(N-2)
        K = SparseMatrix({0: 1/((k+1)*(k+2)*2)}, (N-2, N-2))
        A[0] *= K[0]
        A[2] *= K[0][:-2]
        B[-2] *= K[0][2:]
        B[0] *= K[0]
        B[2] *= K[0][:-2]
        B[4] *= K[0][:-4]

    if returntype == 0:
        M = alpha*B.diags()-A.diags()
        con = np.linalg.cond(M.toarray())

    elif returntype == 1:
        # Use rnd to get the same random numbers for all methods
        buf = rnd.get(N, np.random.random(N))
        if not N in rnd:
            rnd[N] = buf
        v = Function(TS, buffer=buf)
        v[-2:] = 0
        u_hat = Function(TS)
        f_hat = Function(TS)
        f_hat = matvec(v, f_hat, A, B, alpha, method)
        u_hat = solve(f_hat, u_hat, A, B, alpha, method)
        con = np.abs(u_hat-v).max()

    elif returntype == 2:
        fe = alpha*ue - ue.diff(x, 2)
        f_hat = Function(ST)
        fj = Array(ST, buffer=fe)
        if wt != 1:
            fj *= np.sin((np.arange(N)+0.5)*np.pi/N)**2
        f_hat = ST.scalar_product(fj, f_hat, fast_transform=True)

        if method == 4:
            f_hat[:-2] = Q2.diags('csc')*f_hat

        if method == 3:
            f_hat[:-2] *= K[0]

        sol = get_solver(A, B, alpha, method)
        u_hat = Function(TS)
        u_hat = solve(f_hat, u_hat, A, B, alpha, method)
        uj = Array(TS)
        uj = TS.backward(u_hat, uj, fast_transform=True)
        ua = Array(TS, buffer=ue)
        con = np.sqrt(inner(1, (uj-ua)**2))

    return con

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse
    import os
    import sys
    import yaml

    parser = argparse.ArgumentParser(description='Solve the Helmholtz problem with Dirichlet boundary conditions')
    parser.add_argument('--return_type', action='store', type=int, required=True)
    parser.add_argument('--include_legendre', action='store_true')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--numba', action='store_true')
    args = parser.parse_args()

    if args.numba:
        # The numba configuration must be set before importing shenfun.
        try:
            import numba
            cfg = {'optimization': {'mode': 'numba', 'verbose': False}}
            with open('shenfun.yaml', 'w') as f:
                yaml.dump(cfg, f)
            f.close()
        except ModuleNotFoundError:
            os.warning('Numba not found - using Cython')

    cond = []
    if args.return_type == 2:
        N = (2**4,2**6, 2**8, 2**12, 2**16, 2**20)
    elif args.return_type == 1:
        N = (2**4, 2**12, 2**20)
    else:
        N = (32, 64, 128, 256, 512, 1024, 2048)
    M = 6 if args.include_legendre else 5
    alphas = (0, 1000)

    if args.return_type in (0, 2):
        for alpha in alphas:
            cond.append([])
            if args.verbose > 0:
                print('alpha =', alpha)
            for basis in range(M): # To include Legendre use --include_legendre (takes hours for N=2**20)
                if args.verbose > 1:
                    print('Method =', basis)
                cond[-1].append([])
                for n in N:
                    if args.verbose > 2:
                        print('N =', n)
                    cond[-1][-1].append(main(n, basis, alpha, args.return_type))
        linestyle = {0: 'solid', 1: 'dashed', 2: 'dotted'}
        for i in range(len(cond)):
            plt.loglog(N, cond[i][0], 'b',
                       N, cond[i][1], 'r',
                       N, cond[i][2], 'k',
                       N, cond[i][3], 'm',
                       N, cond[i][4], 'y',
                       linestyle=linestyle[i])
            if args.include_legendre:
                plt.loglog(N, cond[i][5], 'y', linestyle=linestyle[i])
            a2l.to_ltx(np.array(cond)[i], frmt='{:6.2e}', print_out=True, mathform=False)
    else:
        for basis in range(M):
            cond.append([])
            if args.verbose > 1:
                print('Method =', basis)

            for alpha in alphas:
                if args.verbose > 0:
                    print('alpha =', alpha)
                for n in N:
                    if args.verbose > 2:
                        print('N =', n)
                    cond[-1].append(main(n, basis, alpha, args.return_type))

        a2l.to_ltx(np.array(cond), frmt='{:6.2e}', print_out=True, mathform=False)
    if os.path.exists('shenfun.yaml'):
        os.remove('shenfun.yaml')
    if args.plot:
        plt.show()
