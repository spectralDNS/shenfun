r"""
Solve Poisson equation on (0, 2pi)x(0, 2pi)x(0, 2pi) with periodic bcs

    \nabla^2 u = f,

Use Fourier basis and find u in VxVxV such that

    (v, div(grad(u))) = (v, f)    for all v in VxVxV

where V is the Fourier basis span{exp(1jkx)}_{k=-N/2}^{N/2-1} and
VxVxV is a tensorproductspace.

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
z = Symbol("z")
u = cos(4*x) + sin(8*y) + sin(6*z)
f = u.diff(x, 2) + u.diff(y, 2) + u.diff(z, 2)

ul = lambdify((x, y, z), u, 'numpy')
fl = lambdify((x, y, z), f, 'numpy')

# Size of discretization
N = 32

K0 = C2CBasis(N)
K1 = C2CBasis(N)
K2 = R2CBasis(N)
T = TensorProductSpace(comm, (K0, K1, K2))
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D

# Get f on quad points
fj = fl(X[0], X[1], X[2])

# Compute right hand side
f_hat = T.scalar_product(fj)

# Solve Poisson equation
v = T.test_function()
A = inner_product(v, div(grad(v)))
f_hat = f_hat / A['diagonal']

uq = T.backward(f_hat, fast_transform=True)

uj = ul(X[0], X[1], X[2])
print(abs(uj-uq).max())
assert np.allclose(uj, uq)

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
#plt.show()

P0 = C2CBasis(N, padding_factor=1.5)
P1 = C2CBasis(N, padding_factor=1.5)
P2 = R2CBasis(N, padding_factor=1.5)
P = TensorProductSpace(comm, (P0, P1, P2))
XP = P.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
up = P.backward(f_hat)
up_hat = P.forward(up)
assert np.allclose(up_hat, f_hat)

plt.figure()
plt.contourf(XP[0][:,:,0], XP[1][:,:,0], up[:, :, 0])
plt.colorbar()
plt.show()
