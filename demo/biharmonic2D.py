r"""
Solve Biharmonic equation in 2D with periodic bcs in one direction
and homogeneous Dirichlet and Neumann in the other

    \nabla^4 u = f,

Use Fourier basis for the periodic direction and Shen's Biharmonic
basis for the non-periodic direction.

"""
import sys
import os
import importlib
from sympy import symbols, cos, sin, chebyshevt
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, Array, \
    Function, TensorProductSpace, FunctionSpace, extract_bc_matrices, comm

# Collect basis and solver from either Chebyshev or Legendre submodules
family = sys.argv[-1].lower() if len(sys.argv) == 2 else 'chebyshev'
base = importlib.import_module('.'.join(('shenfun', family)))
BiharmonicSolver = base.la.Biharmonic

# Use sympy to compute a rhs, given an analytical solution
x, y = symbols("x,y", real=True)
a = 1
b = -1
if family == 'jacobi':
    a = 0
    b = 0
ue = (sin(2*np.pi*x)*cos(2*y))*(1-x**2) + a*(0.5-9./16.*x+1./16.*chebyshevt(3, x)) + b*(0.5+9./16.*x-1./16.*chebyshevt(3, x))
#ue = (sin(2*np.pi*x)*cos(2*y))*(1-x**2) + a*(0.5-0.6*x+1/10*legendre(3, x)) + b*(0.5+0.6*x-1./10.*legendre(3, x))
fe = ue.diff(x, 4) + ue.diff(y, 4) + 2*ue.diff(x, 2, y, 2)

# Size of discretization
N = (30, 30)

if family == 'chebyshev':
    assert N[0] % 2 == 0, "Biharmonic solver only implemented for even numbers"

#SD = FunctionSpace(N[0], family=family, bc='Biharmonic')
SD = FunctionSpace(N[0], family=family, bc=(a, b, 0, 0))
K1 = FunctionSpace(N[1], family='F')
T = TensorProductSpace(comm, (SD, K1), axes=(0, 1))

u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = Array(T, buffer=fe)

# Compute right hand side of biharmonic equation
f_hat = inner(v, fj)

# Get left hand side of biharmonic equation
matrices = inner(v, div(grad(div(grad(u)))))

u_hat = Function(T).set_boundary_dofs() # Solution spectral space

if SD.has_nonhomogeneous_bcs:
    bc_mats = extract_bc_matrices([matrices])
    w0 = np.zeros_like(u_hat)
    for mat in bc_mats:
        f_hat -= mat.matvec(u_hat, w0)

# Create linear algebra solver
H = BiharmonicSolver(*matrices)

# Solve and transform to real space
u_hat = H(u_hat, f_hat)       # Solve

uq = u_hat.backward()

# Compare with analytical solution
uj = Array(T, buffer=ue)
print(abs(uj-uq).max())
assert np.allclose(uj, uq, 1e-8)

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
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
