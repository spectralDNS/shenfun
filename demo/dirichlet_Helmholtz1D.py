r"""
Solve Helmholtz equation in 1D with Dirichlet bcs

    alfa u - \nabla^2 u = f,

The equation to solve for Legendre basis is

    alfa (u, v) + (\nabla u, \nabla v) = (f, v)

whereas for Chebyshev we solve

    alfa (u, v) - (\nabla^2 u, v) = (f, v)

"""
import sys
import importlib
from sympy import symbols, sin, lambdify
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, \
    Array, Function, Basis

assert len(sys.argv) == 3
assert sys.argv[-1].lower() in ('legendre', 'chebyshev')
assert isinstance(int(sys.argv[-2]), int)

# Collect basis and solver from either Chebyshev or Legendre submodules
family = sys.argv[-1].lower()
base = importlib.import_module('.'.join(('shenfun', family)))
Solver = base.la.Helmholtz

# Use sympy to compute a rhs, given an analytical solution
alfa = 1.
x = symbols("x")
ue = sin(4*np.pi*x)*(1-x**2)
fe = alfa*ue - ue.diff(x, 2)

# Lambdify for faster evaluation
ul = lambdify(x, ue, 'numpy')
fl = lambdify(x, fe, 'numpy')

# Size of discretization
N = int(sys.argv[-2])

SD = Basis(N, family=family, bc='Dirichlet', plan=True)
X = SD.mesh(N)
u = TrialFunction(SD)
v = TestFunction(SD)

# Get f on quad points
fj = Array(SD, buffer=fl(X))

# Compute right hand side of Poisson equation
f_hat = Array(SD)
f_hat = inner(v, fj, output_array=f_hat)

# Get left hand side of Poisson equation
if family == 'chebyshev':
    A = inner(v, -div(grad(u)))
    B = inner(v, alfa*u)
else:
    A = inner(grad(v), grad(u))
    B = inner(v, alfa*u)

H = Solver(A, B, A.scale, B.scale)
u_hat = Function(SD)           # Solution spectral space
u_hat = H(u_hat, f_hat)
uj = SD.backward(u_hat)

# Compare with analytical solution
ua = ul(X)

if family == 'chebyshev':
    # Compute L2 error norm using Clenshaw-Curtis integration
    from shenfun import clenshaw_curtis1D
    error = clenshaw_curtis1D((uj-ua)**2, quad=SD.quad)
    print("Error=%2.16e" %(error))

else:
    x, w = SD.points_and_weights(N)
    print("Error=%2.16e" %(np.sqrt(np.sum((uj-ua)**2*w))))

assert np.allclose(uj, ua)
