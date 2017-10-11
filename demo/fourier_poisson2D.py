r"""
Solve Poisson equation on (0, 2pi)x(0, 2pi) with periodic bcs

    \nabla^2 u = f, u(2pi, y) = u(0, y), u(x, 2pi) = u(x, 0)

Use Fourier basis and find u in VxV such that

    (v, div(grad(u))) = (v, f)    for all v in VxV

where V is the Fourier basis span{exp(1jkx)}_{k=-N/2}^{N/2-1} and
VxV is a tensorproductspace.

"""
from sympy import Symbol, cos, sin, exp, lambdify
import numpy as np
import os
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun.tensorproductspace import TensorProductSpace
from shenfun import inner, div, grad, TestFunction, TrialFunction, Function, Array
from mpi4py import MPI
try:
    import matplotlib.pyplot as plt
except:
    plt = None

comm = MPI.COMM_WORLD

# Use sympy to compute a rhs, given an analytical solution
x = Symbol("x")
y = Symbol("y")
ue = cos(4*x) + sin(8*y)
fe = ue.diff(x, 2) + ue.diff(y, 2)

ul = lambdify((x, y), ue, 'numpy')
fl = lambdify((x, y), fe, 'numpy')

# Size of discretization
N = (32, 45)

K0 = C2CBasis(N[0])
K1 = R2CBasis(N[1])
T = TensorProductSpace(comm, (K0, K1))
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = fl(*X)

# Compute right hand side
f_hat = inner(v, fj)

# Solve Poisson equation
#A = inner(v, div(grad(u)))
A = inner(grad(v), grad(u))
f_hat = A.solve(-f_hat)

uq = Array(T, False)
uq = T.backward(f_hat, uq, fast_transform=True)

uj = ul(*X)
assert np.allclose(uj, uq)

if not plt is None and not 'pytest' in os.environ:
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

