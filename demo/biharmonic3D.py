r"""
Solve Biharmonic equation in 3D with periodic bcs in two directions
and homogeneous Dirichlet and Neumann in the remaining third

    \nabla^4 u = f,

Use Fourier basis for the periodic direction and Shen's Biharmonic
basis for the non-periodic direction.

"""
import sys
import os
from sympy import symbols, cos, sin
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, Array, \
    FunctionSpace, TensorProductSpace, Function, comm, la, chebyshev

# Collect basis and solver from either Chebyshev or Legendre submodules
family = sys.argv[-1].lower() if len(sys.argv) == 2 else 'chebyshev'
BiharmonicSolver = chebyshev.la.Biharmonic if family == 'chebyshev' else la.SolverGeneric1ND

# Use sympy to compute a rhs, given an analytical solution
x, y, z = symbols("x,y,z", real=True)
ue = (sin(4*np.pi*x)*sin(6*z)*cos(4*y))*(1-x**2)
fe = ue.diff(x, 4) + ue.diff(y, 4) + ue.diff(z, 4) + 2*ue.diff(x, 2, y, 2) + 2*ue.diff(x, 2, z, 2) + 2*ue.diff(y, 2, z, 2)

# Size of discretization
N = (36, 36, 36)

if family == 'chebyshev':
    assert N[0] % 2 == 0, "Biharmonic solver only implemented for even numbers"

SD = FunctionSpace(N[0], family=family, bc=(0, 0, 0, 0))
K1 = FunctionSpace(N[1], family='F', dtype='D')
K2 = FunctionSpace(N[2], family='F', dtype='d')
T = TensorProductSpace(comm, (SD, K1, K2), axes=(0, 1, 2))

u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = Array(T, buffer=fe)

# Compute right hand side of biharmonic equation
f_hat = inner(v, fj)

# Get left hand side of biharmonic equation
matrices = inner(v, div(grad(div(grad(u)))))

# Create linear algebra solver
H = BiharmonicSolver(matrices)

# Solve and transform to real space
u_hat = Function(T)             # Solution spectral space
u_hat = H(f_hat, u_hat)         # Solve
uq = u_hat.backward()
#uh = uq.forward()

# Compare with analytical solution
uj = Array(T, buffer=ue)
print(abs(uj-uq).max())
assert np.allclose(uj, uq)

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
    plt.contourf(X[0][:, :, 0], X[1][:, :, 0], uq[:, :, 8])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0][:, :, 0], X[1][:, :, 0], uj[:, :, 8])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[0][:, :, 0], X[1][:, :, 0], uq[:, :, 8]-uj[:, :, 8])
    plt.colorbar()
    plt.title('Error')

    #plt.show()
