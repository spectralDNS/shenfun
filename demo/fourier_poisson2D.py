r"""
Solve Poisson equation on (0, 2pi)x(0, 2pi) with periodic bcs

    \nabla^2 u = f, u(2pi, y) = u(0, y), u(x, 2pi) = u(x, 0)

Use Fourier basis

"""
from sympy import Symbol, cos, sin, exp, lambdify
import numpy as np
import matplotlib.pyplot as plt
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun.tensorproductspace import TensorProductSpace, Function,\
    inner_product
from shenfun.operators import div, grad
from mpi4py import MPI

comm = MPI.COMM_WORLD

# Use sympy to compute a rhs, given an analytical solution
x = Symbol("x")
y = Symbol("y")
u = cos(4*x) + sin(8*y)
f = u.diff(x, 2) + u.diff(y, 2)

ul = lambdify((x, y), u, 'numpy')
fl = lambdify((x, y), f, 'numpy')

# Size of discretization
N = (32, 45)

K0 = C2CBasis(N[0])
K1 = R2CBasis(N[1])
T = TensorProductSpace(comm, (K0, K1))
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D

# Get f on quad points
fj = fl(X[0], X[1])

# Compute right hand side
f_hat = T.scalar_product(fj, fast_transform=True)

# Solve Poisson equation
v = T.test_function()
A = inner_product(v, div(grad(v)))
f_hat = f_hat / A['diagonal']

uq = T.backward(f_hat, fast_transform=True)

uj = ul(X[0], X[1])
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
plt.show()

