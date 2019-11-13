r"""
Solve Helmholtz equation in 1D with Dirichlet bcs

.. math::

    \alpha u - \nabla^2 u = f, \quad u(\pm 1) = 0

The equation to solve is

.. math::

    \alpha (u, v)_w - (\nabla^2 u, v)_w = (f, v)_w

The weighted inner product over the domain :math:`\Omega` is defined as

.. math::

    (u, v)_w = \int_{\Omega} u v w dx

where :math:`w(x)` is a weight function.

For either Chebyshev, Legendre or Jacobi we choose a basis that satsifies
the boundary conditions.

"""
import sys
import importlib
from sympy import symbols, sin
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, \
    Array, Function, Basis, dx

assert len(sys.argv) == 3
assert sys.argv[-1].lower() in ('legendre', 'chebyshev', 'jacobi')
assert isinstance(int(sys.argv[-2]), int)

# Collect basis and solver from either Chebyshev or Legendre submodules
family = sys.argv[-1].lower()
base = importlib.import_module('.'.join(('shenfun', family)))
Solver = base.la.Helmholtz

# Use sympy to compute a rhs, given an analytical solution
alfa = 2.
x = symbols("x")
ue = sin(4*np.pi*x)*(1-x**2)
fe = alfa*ue - ue.diff(x, 2)

# Size of discretization
N = int(sys.argv[-2])

SD = Basis(N, family=family, bc='Dirichlet')
X = SD.mesh()
u = TrialFunction(SD)
v = TestFunction(SD)

# Get f on quad points
fj = Array(SD, buffer=fe)

# Compute right hand side of Poisson equation
f_hat = Array(SD)
f_hat = inner(v, fj, output_array=f_hat)

# Get left hand side of Poisson equation
A = inner(v, -div(grad(u)))
B = inner(v, alfa*u)

H = Solver(A, B)
u_hat = Function(SD)           # Solution spectral space
u_hat = H(u_hat, f_hat)
uj = SD.backward(u_hat)

# Compare with analytical solution
ua = Array(SD, buffer=ue)

error = dx((uj-ua)**2)
print('Error=%2.6e'%(np.sqrt(error)))
assert np.allclose(uj, ua)
