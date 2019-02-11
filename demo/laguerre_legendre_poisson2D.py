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
import sys
import os
from sympy import symbols, sin, exp, lambdify
import numpy as np
from mpi4py import MPI
from shenfun import inner, grad, TestFunction, TrialFunction, \
    Array, Function, Basis, TensorProductSpace
from shenfun.la import SolverGeneric2NP

comm = MPI.COMM_WORLD

assert comm.Get_size() == 1, "Two non-periodic directions only have solver implemented for serial"

assert len(sys.argv) == 2, "Call with one command-line arguments"
assert isinstance(int(sys.argv[-1]), int)

# Use sympy to compute a rhs, given an analytical solution
x, y = symbols("x,y")
ue = sin(4*np.pi*y)*sin(2*x)*exp(-x)
fe = ue.diff(x, 2) + ue.diff(y, 2)

# Lambdify for faster evaluation
ul = lambdify((x, y), ue, 'numpy')
fl = lambdify((x, y), fe, 'numpy')

# Size of discretization
N = (int(sys.argv[-1]), int(sys.argv[-1])//2)

D0 = Basis(N[0], 'Laguerre', bc=(0, 0))
D1 = Basis(N[1], 'Legendre', bc=(0, 0))
T = TensorProductSpace(comm, (D0, D1), axes=(0, 1))
X = T.local_mesh(True)
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = Array(T, buffer=fl(*X))

# Compute right hand side of Poisson equation
f_hat = Function(T)
f_hat = inner(v, -fj, output_array=f_hat)

# Get left hand side of Poisson equation
matrices = inner(grad(v), grad(u))

# Create linear algebra solver
H = SolverGeneric2NP(matrices)

# Solve and transform to real space
u_hat = Function(T)           # Solution spectral space
u_hat = H(f_hat, u_hat)       # Solve
uq = u_hat.backward()

# Compare with analytical solution
uj = ul(*X)
assert np.allclose(uj, uq, atol=1e-6)
if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
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

    #plt.show()
