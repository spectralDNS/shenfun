r"""
Solve Poisson equation in 2D with homogeneous Dirichlet bcs in one direction and
periodic in the other. The domain is (-inf, inf) x [0, 2pi]

.. math::

    \nabla^2 u = f,

The equation to solve for a Hermite x Fourier basis is

.. math::

     (\nabla u, \nabla v) = -(f, v)

"""
import os
import sys
from sympy import symbols, exp, hermite, cos
import numpy as np
from shenfun import inner, grad, TestFunction, TrialFunction, la, \
    Array, Function, FunctionSpace, TensorProductSpace, comm

assert len(sys.argv) == 2, 'Call with one command-line argument'
assert isinstance(int(sys.argv[-1]), int)

# Use sympy to compute a rhs, given an analytical solution
x, y = symbols("x,y", real=True)
#ue = sin(4*x)*exp(-x**2)
ue = cos(4*y)*hermite(4, x)*exp(-x**2/2)
fe = ue.diff(x, 2)+ue.diff(y, 2)

# Size of discretization
N = int(sys.argv[-1])

SD = FunctionSpace(N, 'Hermite')
K0 = FunctionSpace(N, 'Fourier', dtype='d')
T = TensorProductSpace(comm, (SD, K0), axes=(0, 1))
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = Array(T, buffer=fe)

# Compute right hand side of Poisson equation
f_hat = Function(T)
f_hat = inner(v, -fj, output_array=f_hat)

# Get left hand side of Poisson equation
matrices = inner(grad(v), grad(u))

# Solve and transform to real space
u_hat = Function(T)           # Solution spectral space
sol = la.SolverGeneric1ND(matrices)
u_hat = sol(f_hat, u_hat)
uq = u_hat.backward()

# Compare with analytical solution
uj = Array(T, buffer=ue)
assert np.allclose(uj, uq, atol=1e-5)
print('Error ', np.linalg.norm(uj-uq))
if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    X = T.local_mesh(True)
    plt.contourf(X[0], X[1], uq)
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0], X[1], uj)
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0], X[1], uq-uj)
    plt.colorbar()
    plt.title('Error')

    plt.figure()
    X = T.local_mesh()
    for x in np.squeeze(X[0]):
        plt.plot((x, x), (np.squeeze(X[1])[0], np.squeeze(X[1])[-1]), 'k')
    for y in np.squeeze(X[1]):
        plt.plot((np.squeeze(X[0])[0], np.squeeze(X[0])[-1]), (y, y), 'k')

    #plt.show()
