r"""
Solve Poisson equation on (0, 2pi)x(0, 2pi)x(0, 2pi) with periodic bcs

    \nabla^2 u = f,

Use Fourier basis and find u in VxVxV such that

    (v, div(grad(u))) = (v, f)    for all v in VxVxV

where V is the Fourier basis span{exp(1jkx)}_{k=-N/2}^{N/2-1} and
VxVxV is a tensorproductspace.

"""
import os
from sympy import symbols, cos, sin, lambdify
import numpy as np
from mpi4py import MPI
from shenfun import inner, div, grad, TestFunction, TrialFunction, Basis, \
    TensorProductSpace, Array, Function, dx

comm = MPI.COMM_WORLD

# Use sympy to compute a rhs, given an analytical solution
x, y, z = symbols("x,y,z")
ue = cos(4*x) + sin(4*y) + sin(6*z)
fe = ue.diff(x, 2) + ue.diff(y, 2) + ue.diff(z, 2)

# Size of discretization
N = (14, 15, 16)

K0 = Basis(N[0], 'F', dtype='D')
K1 = Basis(N[1], 'F', dtype='D')
K2 = Basis(N[2], 'F', dtype='d')
T = TensorProductSpace(comm, (K0, K1, K2))
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = Array(T, buffer=fe)

# Compute right hand side
f_hat = Function(T)
f_hat = inner(v, fj, output_array=f_hat)

# Solve Poisson equation
A = inner(v, div(grad(u)))
f_hat = A.solve(f_hat)

uq = T.backward(f_hat, fast_transform=True)

uj = Array(T, buffer=ue)
print(np.sqrt(dx((uj-uq)**2)))
assert np.allclose(uj, uq)

# Test eval at point
point = np.array([[0.1, 0.5], [0.5, 0.6], [0.1, 0.2]])
p = T.eval(point, f_hat)
ul = lambdify((x, y, z), ue)
assert np.allclose(p, ul(*point))
p2 = f_hat.eval(point)
assert np.allclose(p2, ul(*point))

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.contourf(X[0][:, :, 0], X[1][:, :, 0], uq[:, :, 0])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0][:, :, 0], X[1][:, :, 0], uj[:, :, 0])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0][:, :, 0], X[1][:, :, 0], uq[:, :, 0]-uj[:, :, 0])
    plt.colorbar()
    plt.title('Error')
    plt.show()
