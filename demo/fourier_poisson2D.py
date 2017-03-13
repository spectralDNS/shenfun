r"""
Solve Poisson equation on (0, 2pi)x(0, 2pi) with periodic bcs

    \nabla^2 u = f, u(2pi, y) = u(0, y), u(x, 2pi) = u(x, 0)

Use Fourier basis

"""
from sympy import Symbol, cos, sin, exp, lambdify
import numpy as np
import matplotlib.pyplot as plt
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun.tensorproductspace import TensorProductSpace, Function
import shenfun
from shenfun import inner_product
from mpi4py import MPI

comm = MPI.COMM_WORLD

# Use sympy to compute a rhs, given an analytical solution
x = Symbol("x")
y = Symbol("y")
u = cos(4*x) + sin(8*y)
#u = exp(1j*4*x)
f = u.diff(x, 2) + u.diff(y, 2)

ul = lambdify((x, y), u, 'numpy')
fl = lambdify((x, y), f, 'numpy')

# Size of discretization
N = 32

K0 = C2CBasis(N)
K1 = R2CBasis(N)
T = TensorProductSpace(comm, (K0, K1))
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D

# Get f on quad points
fj = fl(X[0], X[1])

# Compute right hand side
f_hat = T.scalar_product(fj, fast_transform=True)

## Solve Poisson equation

#A0 = inner_product((K0, 0), (K0, 2))
#A1 = inner_product((K1, 0), (K1, 2))

## Modify for 2D. 2pi is the factor from the scalar product in the other direction
#A0[0] = A0[0][:, np.newaxis]*2*np.pi
#A1[0] = A1[0][np.newaxis, :]*2*np.pi
#A = A0+A1
#A[0][0, 0] = 1
#f_hat = f_hat/A[0][T.local_slice(True)]

K = T.local_wavenumbers()
K2 = (K[0]**2+K[1]**2)*(2*np.pi)**2
f_hat = -f_hat / np.where(K2==0, 1., K2)

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
#plt.show()

#plt.figure()
#plt.plot(points, uj)
#plt.title("U")
#plt.figure()
#plt.plot(points, uq - uj)
#plt.title("Error")
#plt.show()

