"""
This script has been used to compute the generalised Chebyshev-Tau Dirichlet results of the paper

    Efficient spectral-Galerkin methods for second-order equations using different Chebyshev bases

The results have been computed using Python 3.9 and dedalus version 2.1905

The script has been created by Keaton Burns.
"""
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, arithmetic
from dedalus.tools.config import config
from dedalus.tools import jacobi

config['matrix construction']['STORE_EXPANDED_MATRICES'] = "True"

def main(N, alpha, method, tau):
    # Bases
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    xb = basis.ChebyshevT(c, size=N, bounds=(-1, 1))
    x = xb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, bases=(xb,), dtype=np.float64)
    ux = field.Field(name='ux', dist=d, bases=(xb,), dtype=np.float64)
    f = field.Field(name='f', dist=d, bases=(xb,), dtype=np.float64)
    t1 = field.Field(name='t1', dist=d, dtype=np.float64)
    t2 = field.Field(name='t2', dist=d, dtype=np.float64)
    pi, sin, cos = np.pi, np.sin, np.cos
    f['g'] = 64*pi**3*(4*pi*sin(4*pi*x)**2*sin(4*pi*cos(4*pi*x)) + cos(4*pi*x)*cos(4*pi*cos(4*pi*x))) + alpha*sin(4*pi*cos(4*pi*x))
    # Tau polynomials
    xb1 = xb._new_a_b(xb.a+1, xb.b+1)
    xb2 = xb._new_a_b(xb.a+2, xb.b+2)
    if tau == 0:
        # First-order classical Chebyshev tau: T[-1], dx(T[-1])
        p1 = field.Field(name='p1', dist=d, bases=(xb,), dtype=np.float64)
        p2 = field.Field(name='p2', dist=d, bases=(xb,), dtype=np.float64)
        p1['c'][-1] = 1
        p2['c'][-1] = 1
    elif tau == 1:
        # First-order ultraspherical tau: U[-1], dx(U[-1])
        p1 = field.Field(name='p1', dist=d, bases=(xb1,), dtype=np.float64)
        p2 = field.Field(name='p2', dist=d, bases=(xb1,), dtype=np.float64)
        p1['c'][-1] = 1
        p2['c'][-1] = 1
    # Problem
    dx = lambda A: operators.Differentiate(A, c)
    problem = problems.LBVP([u, ux, t1, t2])
    problem.add_equation((alpha*u - dx(ux) + t1*p1, f))
    problem.add_equation((ux - dx(u) + t2*p2, 0))
    problem.add_equation((ux(x=-1) + ux(x=+1), 0))
    problem.add_equation((u(x=+1) + u(x=-1), 0))
    solver = solvers.LinearBoundaryValueSolver(problem)
    # Methods
    if method == 0:
        # Condition number
        L_exp = solver.subproblems[0].L_exp
        result = np.linalg.cond(L_exp.A)
    elif method == 1:
        # Roundtrip roundoff with uniform u
        A = solver.subproblems[0].L_exp
        v = np.random.rand(A.shape[1])
        f = A * v
        u = solver.subproblem_matsolvers[solver.subproblems[0]].solve(f)
        result = np.max(np.abs(u-v))
    elif method == 2:
        # Manufactured solution
        from mpi4py_fft import fftw as mpi4py_fftw
        solver.solve()
        ue = np.sin(4*np.pi*np.cos(4*np.pi*x))
        d = mpi4py_fftw.aligned(N, fill=0)
        k = 2*(1 + np.arange((N-1)//2))
        d[::2] = (2./N)/np.hstack((1., 1.-k*k))
        w = mpi4py_fftw.aligned_like(d)
        dct = mpi4py_fftw.dctn(w, axes=(0,), type=3)
        weights = dct(d, w)
        result = np.sqrt(np.sum(weights*(u['g']-ue)**2))
    elif method == 3:
        # Roundtrip roundoff with uniform f
        A = solver.subproblems[0].L_exp
        g = np.random.rand(A.shape[0])
        v = solver.subproblem_matsolvers[solver.subproblems[0]].solve(g)
        f = A * v
        u = solver.subproblem_matsolvers[solver.subproblems[0]].solve(f)
        result = np.max(np.abs(u-v))
    return result

if __name__=='__main__':
    import sys
    import array_to_latex as a2l
    method = int(sys.argv[-2])
    tau = int(sys.argv[-1])
    if method == 0:
        N = (32, 64, 128, 256, 512, 1024, 2048)
    else:
        N = (2**4, 2**8, 2**12, 2**16, 2**20)
    alphas = (0, 1, 1000)
    cond = []
    for alpha in alphas:
        cond.append([])
        for n in N:
            cond[-1].append(main(n, alpha, method, tau))
            print(alpha, n, cond[-1][-1])
    a2l.to_ltx(np.array(cond), frmt='{:6.2e}', mathform=False, print_out=True)

