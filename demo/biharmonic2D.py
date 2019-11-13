r"""
Solve Biharmonic equation in 2D with periodic bcs in one direction
and homogeneous Dirichlet and Neumann in the other

    \nabla^4 u = f,

Use Fourier basis for the periodic direction and Shen's Biharmonic
basis for the non-periodic direction.

"""
import sys
import os
import importlib
from sympy import symbols, cos, sin
import numpy as np
from mpi4py import MPI
from shenfun import inner, div, grad, TestFunction, TrialFunction, Array, \
    Function, TensorProductSpace, Basis

comm = MPI.COMM_WORLD

# Collect basis and solver from either Chebyshev or Legendre submodules
family = sys.argv[-1].lower() if len(sys.argv) == 2 else 'chebyshev'
base = importlib.import_module('.'.join(('shenfun', family)))
BiharmonicSolver = base.la.Biharmonic

# Use sympy to compute a rhs, given an analytical solution
x, y = symbols("x,y")
ue = (sin(2*np.pi*x)*cos(2*y))*(1-x**2)
fe = ue.diff(x, 4) + ue.diff(y, 4) + 2*ue.diff(x, 2, y, 2)

# Size of discretization
N = (30, 30)

if family == 'chebyshev':
    assert N[0] % 2 == 0, "Biharmonic solver only implemented for even numbers"

SD = Basis(N[0], family=family, bc='Biharmonic')
K1 = Basis(N[1], family='F')
T = TensorProductSpace(comm, (SD, K1), axes=(0, 1))
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = Array(T, buffer=fe)

# Compute right hand side of biharmonic equation
f_hat = inner(v, fj)

# Get left hand side of biharmonic equation
matrices = inner(v, div(grad(div(grad(u)))))

# Create linear algebra solver
H = BiharmonicSolver(*matrices)

# Solve and transform to real space
u_hat = Function(T)           # Solution spectral space
u_hat = H(u_hat, f_hat)       # Solve
uq = u_hat.backward()

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
