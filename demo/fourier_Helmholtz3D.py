r"""
Solve Helmholtz equation on (0, 2pi)x(0, 2pi)x(0, 2pi) with periodic bcs

.. math::

    \nabla^2 u + u = f,

Use Fourier basis and find :math:`u` in :math:`V^3` such that

.. math::

    (v, \nabla^2 u + u) = (v, f), \quad \forall v \in V^3

where V is the Fourier basis :math:`span{exp(1jkx)}_{k=-N/2}^{N/2-1}` and
:math:`V^3` is a tensorproductspace.

"""
import os
import numpy as np
from sympy import symbols, cos, sin
from mpi4py import MPI
from shenfun import inner, div, grad, TestFunction, TrialFunction, Basis, \
    TensorProductSpace, Array, Function, dx

comm = MPI.COMM_WORLD

# Use sympy to compute a rhs, given an analytical solution
x, y, z = symbols("x,y,z")
ue = cos(4*x) + sin(4*y) + sin(6*z)
fe = ue.diff(x, 2) + ue.diff(y, 2) + ue.diff(z, 2) + ue

# Size of discretization
N = 16

K0 = Basis(N, 'F', dtype='D')
K1 = Basis(N, 'F', dtype='D')
K2 = Basis(N, 'F', dtype='d')
T = TensorProductSpace(comm, (K0, K1, K2), slab=True)
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = Array(T, buffer=fe)

# Compute right hand side
f_hat = Function(T)
f_hat = inner(v, fj, output_array=f_hat)

# Solve Poisson equation
A = inner(v, u+div(grad(u)))
f_hat = A.solve(f_hat)

uq = T.backward(f_hat, fast_transform=True)

uj = Array(T, buffer=ue)
print(np.sqrt(dx((uj-uq)**2)))
assert np.allclose(uj, uq)

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
    #plt.show()
