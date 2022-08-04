r"""
Solve Poisson equation in 2D with homogeneous Dirichlet on the
domain is [0, inf] x [-1, 1]

.. math::

    \nabla^2 u = f,

Use Legendre basis for the bounded direction and Laguerre for the open.

The equation to solve is

.. math::

     (\nabla u, \nabla v) = -(f, v)

"""
import os
from sympy import symbols, sin, exp
import numpy as np
from shenfun import inner, grad, TestFunction, TrialFunction, \
    Array, Function, FunctionSpace, TensorProductSpace, comm
from shenfun.la import SolverGeneric2ND

# Use sympy to compute a rhs, given an analytical solution
x, y = symbols("x,y", real=True)
ue = sin(4*np.pi*y)*sin(2*x)*exp(-x)
fe = ue.diff(x, 2) + ue.diff(y, 2)

def main(N):
    D0 = FunctionSpace(N, 'Laguerre', bc=(0,))
    D1 = FunctionSpace(N, 'Legendre', bc=(0, 0))
    T = TensorProductSpace(comm, (D0, D1), axes=(0, 1))
    u = TrialFunction(T)
    v = TestFunction(T)

    # Get f on quad points
    fj = Array(T, buffer=fe)

    # Compute right hand side of Poisson equation
    f_hat = Function(T)
    f_hat = inner(v, -fj, output_array=f_hat)

    # Get left hand side of Poisson equation
    matrices = inner(grad(v), grad(u))

    # Create linear algebra solver
    H = SolverGeneric2ND(matrices)

    # Solve and transform to real space
    u_hat = Function(T)           # Solution spectral space
    u_hat = H(f_hat, u_hat)       # Solve
    uq = u_hat.backward()

    # Compare with analytical solution
    uj = Array(T, buffer=ue)
    error = np.sqrt(inner(1, (uj-uq)**2))
    print(f"laguerre_legendre_poisson2D L2 error {error:2.6e}")

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
        plt.figure()
        X = T.local_mesh()
        for x in np.squeeze(X[0]):
            plt.plot((x, x), (np.squeeze(X[1])[0], np.squeeze(X[1])[-1]), 'k')
        for y in np.squeeze(X[1]):
            plt.plot((np.squeeze(X[0])[0], np.squeeze(X[0])[-1]), (y, y), 'k')
        plt.show()
    else:
        assert error < 1e-5

if __name__ == '__main__':
    main(60)
