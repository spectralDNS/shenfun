r"""
Solve Poisson equation in 2D with periodic bcs in one direction
and homogeneous Dirichlet in the other. The domain is [0, inf] x [0, 2\pi]

.. math::

    -\nabla^2 u = f,

Use Fourier basis for the periodic direction and Shen's Dirichlet basis for the
non-periodic direction.

The equation to solve for the Laguerre basis is

.. math::

     -(\nabla^2 u, v) = (f, v)

"""
import os
import sympy as sp
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, la, \
    Array, Function, FunctionSpace, TensorProductSpace, dx, comm

# Use sympy to compute a rhs, given an analytical solution
x, y = sp.symbols("x,y", real=True)
ue = sp.cos(2*y)*sp.cos(2*x)*sp.exp(-x)
fe = -ue.diff(x, 2) - ue.diff(y, 2)

bcs = [
    {'left': {'D': ue.subs(x, 0)}},
    {'left': {'N': sp.diff(ue, x, 1).subs(x, 0)}}
]

def main(N, bc):
    SD = FunctionSpace(N, 'Laguerre', bc=bcs[bc])
    K1 = FunctionSpace(N, 'Fourier', dtype='d')
    T = TensorProductSpace(comm, (SD, K1), axes=(0, 1))
    u = TrialFunction(T)
    v = TestFunction(T)

    # Get f on quad points
    fj = Array(T, buffer=fe)

    # Compute right hand side of Poisson equation
    f_hat = Function(T)
    f_hat = inner(v, fj, output_array=f_hat)

    # Get left hand side of Poisson equation
    matrices = inner(v, -div(grad(u)))

    # Solve and transform to real space
    u_hat = Function(T)           # Solution spectral space
    sol = la.SolverGeneric1ND(matrices)
    u_hat = sol(f_hat, u_hat)
    uq = u_hat.backward()

    # Compare with analytical solution
    uj = Array(T, buffer=ue)
    error = np.sqrt(inner(1, (uj-uq)**2))
    d = {0: 'Dirichlet', 1: 'Neumann'}
    print(f"laguerre_poisson2D {d[bc]:10s} L2 error {error:2.6e}")

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
        assert error < 1e-6
    T.destroy()

if __name__ == '__main__':
    for bc in range(2):
        main(60, bc)
