r"""
Solve Poisson equation on (0, 2pi)x(0, 2pi)x(0, 2pi) with periodic bcs

    \nabla^2 u = f,

Use Fourier basis and find u in VxVxV such that

    (v, div(grad(u))) = (v, f)    for all v in VxVxV

where V is the Fourier basis span{exp(1jkx)}_{k=-N/2}^{N/2-1} and
VxVxV is a tensorproductspace.

"""
from sympy import symbols, cos, sin, exp, lambdify
import numpy as np
import os
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun.tensorproductspace import TensorProductSpace
from shenfun import inner, div, grad, TestFunction, TrialFunction, Function
from mpi4py import MPI
try:
    import matplotlib.pyplot as plt
except:
    plt = None

comm = MPI.COMM_WORLD

# Use sympy to compute a rhs, given an analytical solution
x, y, z = symbols("x,y,z")
ue = cos(4*x) + sin(4*y) + sin(6*z)
fe = ue.diff(x, 2) + ue.diff(y, 2) + ue.diff(z, 2)

ul = lambdify((x, y, z), ue, 'numpy')
fl = lambdify((x, y, z), fe, 'numpy')

# Size of discretization
N = (14, 15, 16)

K0 = C2CBasis(N[0])
K1 = C2CBasis(N[1])
K2 = R2CBasis(N[2])
T = TensorProductSpace(comm, (K0, K1, K2))
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = fl(*X)

# Compute right hand side
f_hat = inner(v, fj)

# Solve Poisson equation
A = inner(v, div(grad(u)))
f_hat = A.solve(f_hat)

uq = T.backward(f_hat, fast_transform=True)

uj = ul(*X)
print(abs(uj-uq).max())
assert np.allclose(uj, uq)

if not plt is None and not 'pytest' in os.environ:
    plt.figure()
    plt.contourf(X[0][:,:,0], X[1][:,:,0], uq[:, :, 0])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0][:,:,0], X[1][:,:,0], uj[:, :, 0])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0][:,:,0], X[1][:,:,0], uq[:, :, 0]-uj[:, :, 0])
    plt.colorbar()
    plt.title('Error')
    plt.show()
