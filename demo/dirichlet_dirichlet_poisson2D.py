r"""
Solve Helmholtz equation in 2D with homogeneous Dirichlet boundary conditions

    au - \nabla^2 u = f,

Use Shen's Legendre Dirichlet basis

The equation to solve is

     a(u, v) + (\nabla u, \nabla v) = (f, v)

"""
import os
import importlib
from sympy import symbols, cos, sin, lambdify
import numpy as np
from mpi4py import MPI
from shenfun import inner, grad, TestFunction, TrialFunction, Function, Basis, \
    TensorProductSpace, Array

comm = MPI.COMM_WORLD

basis = 'legendre'
base = importlib.import_module('.'.join(('shenfun', basis)))
Solver = base.la.Helmholtz_2dirichlet

# Use sympy to compute a rhs, given an analytical solution
a = 2.
x, y = symbols("x,y")
ue = (cos(4*y)*sin(2*x))*(1-x**2)*(1-y**2)
fe = a*ue - ue.diff(x, 2) - ue.diff(y, 2)

# Lambdify for faster evaluation
ul = lambdify((x, y), ue, 'numpy')
fl = lambdify((x, y), fe, 'numpy')

# Size of discretization
N = (32, 34)

SD0 = Basis(N[0], 'L', bc=(0, 0), scaled=True)
SD1 = Basis(N[1], 'L', bc=(0, 0), scaled=True)
T = TensorProductSpace(comm, (SD0, SD1))
X = T.local_mesh(True)
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = Array(T, buffer=fl(*X))

# Compute right hand side of Poisson equation
f_hat = inner(v, fj)

# Get left hand side of Poisson equation
matrices = inner(grad(v), grad(u))
matrices += inner(v, a*u)

# Create Helmholtz linear algebra solver
H = Solver(T, matrices)

# Solve and transform to real space
u_hat = Function(T)           # Solution spectral space
u_hat = H(u_hat, f_hat, 0)    # Solve

uq = Array(T)
uq = T.backward(u_hat, uq)

# Compare with analytical solution
uj = ul(*X)
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
