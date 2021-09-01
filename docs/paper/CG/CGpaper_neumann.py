"""
This script has been used to compute the Neumann results of the paper

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
import array_to_latex as a2l

x = sp.Symbol('x', real=True)

fe = {}

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
    from shenfun import chebyshev
    if method == 2:
        if alpha == 0:
            f_hat = A.matvec(-u_hat, f_hat)

        else:
            sol = chebyshev.la.Helmholtz(A, B, -1, alpha)
            f_hat = sol.matvec(u_hat, f_hat)
    else:
        if alpha == 0:
            f_hat[:-2] = A.diags() * u_hat[:-2]
            f_hat *= -1
        else:
            M = alpha*B - A
            f_hat[:-2] = M.diags() * u_hat[:-2]
    f_hat[0] = 0
    return f_hat

def solve(f_hat, u_hat, A, B, alpha, method):
    """Solve problem for given method

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
    from shenfun import chebyshev, la
    constraints = ((0, 0),) if alpha == 0 else ()

    if method == 2:
        if alpha == 0:
            sol = la.Solver(A)
            f_hat *= -1

        else:
            sol = chebyshev.la.Helmholtz(A, B, -1, alpha)

    elif method in (0, 1, 3, 4):
        if alpha == 0:
            sol = la.TwoDMA(A)
            f_hat *= -1
        else:
            sol = la.FDMA(alpha*B-A)

    elif method == 5:
        if alpha == 0:
            sol = la.Solver(A)
            f_hat *= -1
        else:
            sol = la.TDMA(alpha*B-A)

    else:
        sol = la.Solver(alpha*B-A)

    u_hat = sol(f_hat, u_hat, constraints=constraints)
    u_hat[0] = 0
    return u_hat

def main(N, method=0, alpha=0, returntype=0):
    from shenfun import FunctionSpace, TrialFunction, TestFunction, \
        inner, div, grad, chebyshev, SparseMatrix, Function, Array
    global fe

    basis = {0: ('ShenNeumann', 'CombinedShenNeumann'),
             1: ('ShenDirichlet', 'MikNeumann'),
             2: ('ShenNeumann', 'ShenNeumann'),
             3: ('DirichletU', 'ShenNeumann'),
             4: ('Orthogonal', 'ShenNeumann'),  # Quasi-Galerkin
             5: ('ShenNeumann', 'ShenNeumann'), # Legendre
             }

    test, trial = basis[method]

    wt = {0: 1, 1: 1, 2: 1, 3: 1-x**2, 4: 1, 5: 1}[method]
    family = 'C' if method < 5 else 'L'
    test = FunctionSpace(N, family, basis=test)
    trial = FunctionSpace(N, family, basis=trial)

    v = TestFunction(test)
    u = TrialFunction(trial)
    A = inner(v*wt, div(grad(u)))
    B = inner(v*wt, u)

    if method == 4:
        # Quasi preconditioning
        Q2 = chebyshev.quasi.QIGmat(N)
        A = Q2*A
        B = Q2*B

    if method == 3:
        k = np.arange(N-2)
        K = SparseMatrix({0: 1/(2*(k+1)*(k+2))}, (N-2, N-2))
        A[0] *= K[0]
        A[2] *= K[0][:-2]
        B[-2] *= K[0][2:]
        B[0] *= K[0]
        B[2] *= K[0][:-2]
        B[4] *= K[0][:-4]


    if returntype == 0:
        if alpha == 0:
            con = np.linalg.cond(A.diags().toarray()[1:, 1:])
        else:
            con = np.linalg.cond(alpha*B.diags().toarray()-A.diags().toarray())

    elif returntype == 1:
        v = Function(trial, buffer=np.random.random(N))
        v[0] = 0
        v[-2:] = 0
        u_hat = Function(trial)
        f_hat = Function(trial)
        f_hat = matvec(v, f_hat, A, B, alpha, method)
        u_hat = solve(f_hat, u_hat, A, B, alpha, method)
        con = np.abs(u_hat-v).max()

    elif returntype == 2:
        ue = sp.cos(100*sp.pi*x)
        fe = alpha*ue-ue.diff(x, 2)
        f_hat = Function(test)
        fj = Array(test, buffer=fe)
        if wt != 1:
            if test.quad == 'GC':
                fj *= np.sin((np.arange(N)+0.5)*np.pi/N)**2
            else:
                fj *= np.sin((np.arange(N)+1)*np.pi/(N+1))**2
        f_hat = test.scalar_product(fj, f_hat, fast_transform=True)
        if method == 4:
            f_hat[:-2] = Q2.diags('csc')*f_hat

        if method == 3:
            f_hat[:-2] *= K[0]

        u_hat = Function(trial)
        u_hat = solve(f_hat, u_hat, A, B, alpha, method)
        uj = Array(trial)
        uj = trial.backward(u_hat, uj, fast_transform=True)
        ua = Array(trial, buffer=ue)
        xj, wj = trial.points_and_weights()

        if family == 'C':
            ua -= np.sum(ua*wj)/np.pi # normalize
            uj -= np.sum(uj*wj)/np.pi # normalize

        else:
            ua -= np.sum(ua*wj)/2 # normalize
            uj -= np.sum(uj*wj)/2 # normalize

        con = np.sqrt(inner(1, (uj-ua)**2))
        #con = np.max(abs(uj-ua))
    return con

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse
    import os
    import sys
    import yaml

    parser = argparse.ArgumentParser(description='Solve the Helmholtz problem with Neumann boundary conditions')
    parser.add_argument('--return_type', action='store', type=int, required=True)
    parser.add_argument('--include_legendre', action='store_true')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--numba', action='store_true')
    args = parser.parse_args()

    if args.numba:
        try:
            import numba
            cfg = {'optimization': {'mode': 'numba', 'verbose': False}}
            with open('shenfun.yaml', 'w') as f:
                yaml.dump(cfg, f)
            f.close()
        except ModuleNotFoundError:
            os.warning('Numba not found - using Cython')
    cond = []
    M = 6 if args.include_legendre else 5
    if args.return_type == 2:
        N = (2**4,2**6, 2**8, 2**12, 2**16, 2**20)
    elif args.return_type == 1:
        N = (2**4, 2**12, 2**20)
    else:
        N = (32, 64, 128, 256, 512, 1024, 2048)
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
