r"""
Solve Poisson equation in 2D with homogeneous Dirichlet boundary conditions

    \nabla^2 u = f,

Use Shen's Legendre Dirichlet basis

The equation to solve is

     (\nabla u, \nabla v) = -(f, v)

"""
import sys
import importlib
from sympy import symbols, cos, sin, exp, lambdify
import numpy as np
import matplotlib.pyplot as plt
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun.tensorproductspace import TensorProductSpace
from shenfun import inner, div, grad, TestFunction, TrialFunction, Function, \
    project
from mpi4py import MPI

comm = MPI.COMM_WORLD

basis = 'legendre'
shen = importlib.import_module('.'.join(('shenfun', basis)))
Basis = shen.bases.ShenDirichletBasis
Solver = shen.la.Helmholtz_2dirichlet

# Use sympy to compute a rhs, given an analytical solution
x, y = symbols("x,y")
ue = (cos(4*y) + sin(2*x))*(1-x**2)*(1-y**2)
fe = ue.diff(x, 2) + ue.diff(y, 2)

# Lambdify for faster evaluation
ul = lambdify((x, y), ue, 'numpy')
fl = lambdify((x, y), fe, 'numpy')

# Size of discretization
N = (32, 32)

SD0 = Basis(N[0])
SD1 = Basis(N[1])
T = TensorProductSpace(comm, (SD0, SD1), axes=(0, 1))
X = T.local_mesh(True)
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = fl(*X)

# Compute right hand side of Poisson equation
f_hat = inner(v, -fj)

# Get left hand side of Poisson equation
matrices = inner(grad(v), grad(u))

# Create Helmholtz linear algebra solver
H = Solver(T, matrices)

# Solve and transform to real space
u_hat = Function(T)           # Solution spectral space
u_hat = H(u_hat, f_hat)       # Solve
uq = Function(T, False)
uq = T.backward(u_hat, uq)

# Compare with analytical solution
uj = ul(*X)
print(abs(uj-uq).max())
assert np.allclose(uj, uq)

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

#plt.show()
