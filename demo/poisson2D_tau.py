r"""
Solve Poisson equation in 2D with the tau method

.. math::

    \nabla^2 u = f,

The equation to solve is

.. math::

     (\nabla^2 u, v) = (f, v)

with Dirichlet boundary conditions u(-1, y) = a and u(1, y) = b.

"""
import sys
import sympy as sp
import numpy as np
import scipy.sparse as scp
from shenfun import *

assert len(sys.argv) == 3, 'Call with two command-line arguments'
assert sys.argv[-1].lower() in ('legendre', 'chebyshev')
assert isinstance(int(sys.argv[-2]), int)

# Get family from args
family = sys.argv[-1].lower()

# Size of discretization
N = int(sys.argv[-2])

T0 = FunctionSpace(N, family=family, domain=(-1, 2))
F = FunctionSpace(N, 'F', dtype='d')
T = TensorProductSpace(comm, (T0, F))
u = TrialFunction(T)
v = TestFunction(T)

# Use sympy to compute a rhs, given an analytical solution
x, y = sp.symbols("x,y", real=True)
x_map = T0.map_reference_domain(x)
ue = sp.cos(1*sp.pi*x_map)*sp.sin(4*y)
fe = ue.diff(x, 2) + ue.diff(y, 2)

# Get f on quad points
fj = Array(T, buffer=fe)

# Compute right hand side of Poisson equation
f_hat = Function(T)
f_hat = inner(v, fj, output_array=f_hat)

# Get left hand side of Poisson equation
M = inner(v, div(grad(u)), level=2)

# Fix boundary conditions
A = M[0].mats[0]
B = M[1].mats[0]
A = A.diags('lil')
A[-2] = np.ones(N)
A[-1] = (-1)**np.arange(N)
A = A.tocsc()
B = B.diags('csc')
S = scp.kron(A, M[0].mats[1].diags('csc')) + scp.kron(B, M[1].mats[1].diags('csc'))

# Fix right hand side boundary conditions
f_hat[-2] = project(ue.subs(x, T0.domain[1]), F)
f_hat[-1] = project(ue.subs(x, T0.domain[0]), F)

# Solve
u_hat = Function(T)
u_hat[:] = scp.linalg.spsolve(S, f_hat.flatten()).reshape(u_hat.shape)
uj = u_hat.backward()

# Compare with analytical solution
ua = Array(T, buffer=ue)
print("Error=%2.16e" %(np.sqrt(dx((uj-ua)**2))))
assert np.allclose(uj, ua)
