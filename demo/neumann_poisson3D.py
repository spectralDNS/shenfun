r"""
Solve Poisson equation in 3D with periodic bcs in two directions
and homogeneous Neumann in the third

    \nabla^2 u = f,

Use Fourier basis for the periodic directions and Shen's Neumann basis for the
non-periodic direction.

The equation to solve is

     (\nabla^2 u, v) = (f, v)

"""
import sys
import os
import importlib
from sympy import symbols, cos, sin, lambdify
import numpy as np
from mpi4py import MPI
from shenfun import inner, div, grad, TestFunction, TrialFunction, Array, \
    Function, TensorProductSpace, Basis

comm = MPI.COMM_WORLD

# Collect basis and solver from either Chebyshev or Legendre submodules
family = sys.argv[-1].lower() if len(sys.argv) == 2 else 'chebyshev'
base = importlib.import_module('.'.join(('shenfun', family)))
Solver = base.la.Helmholtz

# Use sympy to compute a rhs, given an analytical solution
x, y, z = symbols("x,y,z")
ue = sin(6*z)*cos(4*y)*sin(2*np.pi*x)*(1-x**2)
fe = ue.diff(x, 2) + ue.diff(y, 2) + ue.diff(z, 2)

# Size of discretization
N = (32, 32, 32)

SD = Basis(N[0], family=family, bc='Neumann')
K1 = Basis(N[1], family='F', dtype='D')
K2 = Basis(N[2], family='F', dtype='d')
T = TensorProductSpace(comm, (SD, K1, K2))
X = T.local_mesh(True)
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
u = T.backward(u_hat)

# Compare with analytical solution
uj = Array(T, buffer=ue)
print(abs(uj-u).max())
assert np.allclose(uj, u)

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.contourf(X[0][:, :, 0], X[1][:, :, 0], u[:, :, 2])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0][:, :, 0], X[1][:, :, 0], uj[:, :, 2])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0][:, :, 0], X[1][:, :, 0], u[:, :, 2]-uj[:, :, 2])
    plt.colorbar()
    plt.title('Error')

    plt.show()
