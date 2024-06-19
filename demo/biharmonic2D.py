r"""
Solve Biharmonic equation in 2D with periodic bcs in one direction
and homogeneous Dirichlet and Neumann in the other

    \nabla^4 u = f,

Use Fourier basis for the periodic direction and Shen's Biharmonic
basis for the non-periodic direction.

"""
import os
import sympy as sp
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, Array, \
    Function, TensorProductSpace, FunctionSpace, comm, la, chebyshev

# Use sympy to compute a rhs, given an analytical solution
x, y = sp.symbols("x,y", real=True)
ue = x**4*sp.sin(2*y)
fe = ue.diff(x, 4) + ue.diff(y, 4) + 2*ue.diff(x, 2, y, 2)

def main(N, family):
    bcs = {'left': {'D': ue.subs(x, -1), 'N': ue.diff(x, 1).subs(x, -1)},
           'right': {'D': ue.subs(x, 1), 'N': ue.diff(x, 1).subs(x, 1)}}
    SD = FunctionSpace(N, family=family, bc=bcs)
    K1 = FunctionSpace(N, family='F')
    T = TensorProductSpace(comm, (SD, K1), axes=(0, 1))

    u = TrialFunction(T)
    v = TestFunction(T)

    # Get f on quad points
    fj = Array(T, buffer=fe)

    # Compute right hand side of biharmonic equation
    f_hat = inner(v, fj)

    # Get left hand side of biharmonic equation
    matrices = inner(v, div(grad(div(grad(u)))))

    u_hat = Function(T) # Solution spectral space

    # Create linear algebra solver
    BiharmonicSolver = chebyshev.la.Biharmonic if SD.family() == 'chebyshev' and N % 2 == 0 else la.SolverGeneric1ND
    H = BiharmonicSolver(matrices)

    # Solve and transform to real space
    u_hat = H(f_hat, u_hat)

    uq = u_hat.backward()

    # Compare with analytical solution
    uj = Array(T, buffer=ue)
    error = np.sqrt(inner(1, (uj-uq)**2))
    if comm.Get_rank() == 0:
        print(f"biharmonic2D {SD.family():14s} L2 error = {error:2.6e}")

    if 'pytest' not in os.environ:
        import matplotlib.pyplot as plt
        plt.figure()
        X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
        plt.contourf(X[0], X[1], uq)
        plt.colorbar()

        plt.figure()
        plt.contourf(X[0], X[1], uj)
        plt.colorbar()

        plt.figure()
        plt.contourf(X[0], X[1], uq-uj)
        plt.colorbar()
        plt.title('Error')
        plt.show()

    else:
        assert error < 1e-6
    
    T.destroy()

if __name__ == '__main__':
    for family in ('legendre', 'chebyshev', 'chebyshevu', 'ultraspherical', 'jacobi'):
        main(24, family)
