r"""
Solve Helmholtz equation in 2D with homogeneous Dirichlet boundary conditions

    au - \nabla^2 u = f,

Use Shen's Dirichlet basis for either Chebyshev, Legendre or Jacobi polynomials.

"""
import os
import sys
import importlib
from sympy import symbols, cos, sin
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, Function, FunctionSpace, \
    TensorProductSpace, Array, comm
from shenfun.la import SolverGeneric2ND

# Use sympy to compute a rhs, given an analytical solution
a = 1.
x, y = symbols("x,y", real=True)
ue = (cos(4*y)*sin(2*x))*(1-x**2)*(1-y**2)
fe = a*ue - ue.diff(x, 2) - ue.diff(y, 2)

def main(N, family):
    assert len(N) == 2
    if family == 'legendre':
        base = importlib.import_module('.'.join(('shenfun', family)))
        Solver = base.la.Helmholtz_2dirichlet
    else:
        Solver = SolverGeneric2ND

    SD0 = FunctionSpace(N[0], family, bc=(0, 0), scaled=True)
    SD1 = FunctionSpace(N[1], family, bc=(0, 0), scaled=True)

    T = TensorProductSpace(comm, (SD0, SD1), axes=(1, 0))
    u = TrialFunction(T)
    v = TestFunction(T)

    # Get f on quad points
    fj = Array(T, buffer=fe)

    # Compute right hand side of Poisson equation
    f_hat = Function(T)
    f_hat = inner(v, fj, output_array=f_hat)

    # Get left hand side of Poisson equation
    matrices = inner(v, -div(grad(u)))
    matrices += [inner(v, a*u)]

    # Create Helmholtz linear algebra solver
    H = Solver(matrices)

    # Solve and transform to real space
    u_hat = Function(T)           # Solution spectral space
    u_hat = H(f_hat, u_hat)    # Solve

    uq = Array(T)
    uq = T.backward(u_hat, uq)

    # Compare with analytical solution
    uj = Array(T, buffer=ue)
    error = np.sqrt(inner(1, (uj-uq)**2))
    print(f'dirichlet_dirichlet_poisson2D {family:14s} L2 error {error:2.6e}')

    if family == 'legendre':
        H.destroy()

    if 'pytest' not in os.environ:
        import matplotlib.pyplot as plt
        plt.figure()
        X = T.local_mesh(True)
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
    for family in ('chebyshev', 'chebyshevu', 'legendre', 'ultraspherical', 'jacobi'):
        main((24, 25), family)
