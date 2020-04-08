r"""
Solve Poisson equation on the 4-dimensional (0, 2pi)^4 with periodic bcs

    \nabla^2 u = f,

Use Fourier basis and find u in V^4 such that

    (v, div(grad(u))) = (v, f)    for all v in V^4

where V is the Fourier space span{exp(1jkx)}_{k=-N/2}^{N/2-1} and
V^4 is a 4-dimensional tensorproductspace.

"""
import os
from sympy import symbols, cos, sin, lambdify
import numpy as np
from mpi4py import MPI
from shenfun import inner, div, grad, TestFunction, TrialFunction, Basis, \
    TensorProductSpace, Array, Function, dx

comm = MPI.COMM_WORLD

# Use sympy to compute a rhs, given an analytical solution
x, y, z, r = symbols("x,y,z,r")
ue = cos(4*x) + sin(4*y) + sin(6*z) + cos(6*r)
fe = ue.diff(x, 2) + ue.diff(y, 2) + ue.diff(z, 2) + ue.diff(r, 2)

# Size of discretization
N = (8, 10, 12, 14)

K0 = Basis(N[0], 'F', dtype='D')
K1 = Basis(N[1], 'F', dtype='D')
K2 = Basis(N[2], 'F', dtype='D')
K3 = Basis(N[3], 'F', dtype='d')
T = TensorProductSpace(comm, (K0, K1, K2, K3))
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

uq = T.backward(f_hat)

uj = Array(T, buffer=ue)
print(np.sqrt(dx((uj-uq)**2)))
assert np.allclose(uj, uq)

print(f_hat.commsizes, fj.commsizes)

if 'pytest' not in os.environ and comm.Get_size() == 1:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.contourf(X[0][:, :, 0, 0], X[1][:, :, 0, 0], uq[:, :, 0, 0])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0][:, :, 0, 0], X[1][:, :, 0, 0], uj[:, :, 0, 0])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0][:, :, 0, 0], X[1][:, :, 0, 0], uq[:, :, 0, 0]-uj[:, :, 0, 0])
    plt.colorbar()
    plt.title('Error')
    plt.show()
