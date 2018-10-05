r"""
Solve Poisson equation on (0, 2pi)x(0, 2pi) with periodic bcs

    \nabla^2 u = f, u(2pi, y) = u(0, y), u(x, 2pi) = u(x, 0)

Use Fourier basis and find u in VxV such that

    (v, div(grad(u))) = (v, f)    for all v in VxV

where V is the Fourier basis span{exp(1jkx)}_{k=-N/2}^{N/2-1} and
VxV is a tensorproductspace.

"""
import os
from sympy import Symbol, cos, sin, lambdify
import numpy as np
from shenfun import inner, grad, TestFunction, TrialFunction, Array, Basis, \
    TensorProductSpace, Function
from mpi4py import MPI
try:
    import matplotlib.pyplot as plt
except ImportError:
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
N = (64, 64)

K0 = Basis(N[0], family='F', dtype='D', domain=(-2*np.pi, 2*np.pi))
K1 = Basis(N[1], family='F', dtype='d', domain=(-2*np.pi, 2*np.pi))
T = TensorProductSpace(comm, (K0, K1), axes=(0, 1))
X = T.local_mesh(True)
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = Array(T, buffer=fl(*X))

# Compute right hand side
f_hat = Function(T)
f_hat = inner(v, fj, output_array=f_hat)

# Solve Poisson equation
u_hat = Function(T)
#A = inner(v, div(grad(u)))
A = inner(grad(v), grad(u))
u_hat = A.solve(-f_hat, u_hat)

uq = Array(T)
uq = T.backward(u_hat, uq, fast_transform=True)

uj = ul(*X)
assert np.allclose(uj, uq)

#from shenfun.tensorproductspace import Convolve
#S0 = Basis(N[0], family='F', dtype='D', padding_factor=2.0)
#S1 = Basis(N[1], family='F', dtype='d', padding_factor=2.0)
#Tp = TensorProductSpace(comm, (S0, S1), axes=(0, 1))
#C0 = Convolve(Tp)
#ff_hat = C0(f_hat, f_hat)

# Test eval at point
point = np.array([[0.1, 0.5], [0.5, 0.6]])
p = T.eval(point, u_hat)
assert np.allclose(p, ul(*point))
p2 = u_hat.eval(point)
assert np.allclose(p2, ul(*point))

if plt is not None and not 'pytest' in os.environ:
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
