r"""
Solve Poisson equation in 2D with periodic bcs in one direction
and homogeneous Neumann in the other

    \nabla^2 u = f,

Use Fourier basis for the periodic direction and Shen's Neumann basis for the
non-periodic direction.

The equation to solve is

     (\nabla^2 u, v) = (f, v)

"""
import sys
import os
import importlib
from sympy import symbols, cos, sin
import numpy as np
from mpi4py import MPI
from shenfun import inner, div, grad, TestFunction, TrialFunction, \
    TensorProductSpace, Basis, Array, Function

comm = MPI.COMM_WORLD

# Collect basis and solver from either Chebyshev or Legendre submodules
family = sys.argv[-1].lower() if len(sys.argv) == 2 else 'chebyshev'
base = importlib.import_module('.'.join(('shenfun', family)))
Solver = base.la.Helmholtz

# Use sympy to compute a rhs, given an analytical solution
x, y = symbols("x,y")
ue = cos(5*y)*(sin(2*np.pi*x))*(1-x**2)
fe = ue.diff(x, 2) + ue.diff(y, 2)

# Size of discretization
N = (31, 32)

SD = Basis(N[0], family=family, bc='Neumann')
K1 = Basis(N[1], family='F', dtype='d')
T = TensorProductSpace(comm, (SD, K1))
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = Array(T, buffer=fe)

# Compute right hand side of Poisson equation
f_hat = inner(v, fj)

# Get left hand side of Poisson equation
matrices = inner(v, div(grad(u)))

# Create Helmholtz linear algebra solver
H = Solver(*matrices)

# Solve and transform to real space
u_hat = Function(T)           # Solution spectral space
u_hat = H(u_hat, f_hat)       # Solve
uq = T.backward(u_hat)

# Compare with analytical solution
uj = Array(T, buffer=ue)
print(abs(uj-uq).max())
assert np.allclose(uj, uq)

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

    plt.show()
