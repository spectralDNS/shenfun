r"""
Solve Poisson equation in 3D with periodic bcs in two directions
and homogeneous Dirichlet in the third

    \nabla^2 u = f,

Use Fourier basis for the periodic directions and Shen's Dirichlet basis for the
remaining non-periodic direction. Discretization leads to a Holmholtz problem.

Note that the equation to solve is

     (\nabla^2 u, v) = (f, v)

"""
import sys
import os
import time
from sympy import symbols, cos, sin
import numpy as np
from mpi4py import MPI
from shenfun import inner, div, grad, TestFunction, TrialFunction, Array, \
    Function, FunctionSpace, TensorProductSpace, dx, comm, la, chebyshev
from mpi4py_fft.pencil import Subcomm

assert len(sys.argv) == 3
assert sys.argv[-1].lower() in ('legendre', 'chebyshev', 'jacobi', 'chebyshevu')
assert isinstance(int(sys.argv[-2]), int)

family = sys.argv[-1].lower()
Solver = chebyshev.la.Helmholtz if family == 'chebyshev' else la.SolverGeneric1ND
regtest = True

# Use sympy for manufactured solution
a = -1
b = 1
if family == 'jacobi':
    a = 0
    b = 0
x, y, z = symbols("x,y,z", real=True)
ue = (cos(4*x) + sin(2*y) + sin(4*z))*(1-y**2) + a*(1 - y)/2. + b*(1 + y)/2.

# Size of discretization
N = int(sys.argv[-2])
N = [N, N+1, N+2]
#N = (14, 15, 16)

SD = FunctionSpace(N[1], family=family, bc=(a, b))
K1 = FunctionSpace(N[0], family='F', dtype='D')
K2 = FunctionSpace(N[2], family='F', dtype='d')
subcomms = Subcomm(MPI.COMM_WORLD, [0, 0, 1])
T = TensorProductSpace(subcomms, (K1, SD, K2), axes=(1, 0, 2))
u = TrialFunction(T)
v = TestFunction(T)

# Get manufactured right hand side
fe = div(grad(u)).tosympy(basis=ue)

K = T.local_wavenumbers()

# Get f on quad points
fj = Array(T, buffer=fe)

# Compute right hand side of Poisson equation
f_hat = inner(v, fj)

# Get left hand side of Poisson equation
matrices = inner(v, div(grad(u)))

# Create Helmholtz linear algebra solver
H = Solver(matrices)

# Solve and transform to real space
u_hat = Function(T)           # Solution spectral space
t0 = time.time()
u_hat = H(f_hat, u_hat)       # Solve
uq = u_hat.backward()

# Compare with analytical solution
uj = Array(T, buffer=ue)
error = comm.reduce(dx((uj-uq)**2))
if comm.Get_rank() == 0 and regtest is True:
    print("Error=%2.16e" %(np.sqrt(error)))
assert np.allclose(uj, uq)

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    X = T.local_mesh()
    plt.contourf(X[2][0, 0, :], X[0][:, 0, 0], uq[:, 2, :])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[2][0, 0, :], X[0][:, 0, 0], uj[:, 2, :])
    plt.colorbar()

    plt.figure()
    plt.contourf(X[2][0, 0, :], X[0][:, 0, 0], uq[:, 2, :]-uj[:, 2, :])
    plt.colorbar()
    plt.title('Error')

    #plt.show()
