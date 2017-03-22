r"""
Solve biharmonic equation in 1D

    u'''' + a*u'' + b*u = f,

Use Shen's Biharmonic basis.

"""
import sys
import importlib
from sympy import symbols, cos, sin, exp, lambdify
import numpy as np
import matplotlib.pyplot as plt
from shenfun import inner_product
from shenfun.operators import div, grad, Dx

# Collect basis and solver from either Chebyshev or Legendre submodules
basis = sys.argv[-1] if len(sys.argv) == 2 else 'chebyshev'
shen = importlib.import_module('.'.join(('shenfun', basis)))
Basis = shen.bases.ShenBiharmonicBasis
Solver = shen.la.Biharmonic

# Use sympy to compute a rhs, given an analytical solution
x = symbols("x")
u = sin(np.pi*x)*(1-x**2)

k = 8
nu = 1./590.
dt = 5e-5
a = -(k**2+nu*dt/2*k**4)
b = 1.0
c = -nu*dt/2.
f = a*u.diff(x, 4) + b*u.diff(x, 2) + c*u

# Lambdify for faster evaluation
ul = lambdify(x, u, 'numpy')
fl = lambdify(x, f, 'numpy')

# Size of discretization
N = 32

SD = Basis(N, plan=True)
X = SD.mesh(N)

# Get f on quad points
fj = fl(X)

# Compute right hand side of Poisson equation
f_hat = np.zeros(N)
f_hat = SD.scalar_product(fj, f_hat)

# Get left hand side of Poisson equation (no integration by parts)
v = SD.test_function()
S = inner_product(v, Dx(v, 0, 4))
A = inner_product(v, Dx(v, 0, 2))
B = inner_product(v, v)

# Create Helmholtz linear algebra solver
H = Solver(S, A, B, a, b, c)

# Solve and transform to real space
u = np.zeros(N)               # Solution real space
u_hat = np.zeros(N)           # Solution spectral space
u_hat = H(u_hat, f_hat)       # Solve
u = SD.backward(u_hat, u)

# Compare with analytical solution
uj = ul(X)
print(abs(uj-u).max())
assert np.allclose(uj, u)

plt.figure()
plt.plot(X, u)

plt.figure()
plt.plot(X, uj)

plt.figure()
plt.plot(X, u-uj)
plt.title('Error')
#plt.show()
