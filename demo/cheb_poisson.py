#pylint: disable=invalid-name
r"""
Solve Poisson equation on (-1, 1) with homogeneous bcs

    \nabla^2 u = f, u(\pm 1) = 0

Use Shen basis \phi_k = T_k - T_{k+2}, where T_k is k'th Chebyshev
polynomial of first kind. Solve using spectral Galerkin and the
weighted L_w norm (u, v)_w = \int_{-1}^{1} u v / \sqrt(1-x^2) dx

    (\nabla^2 u, \phi_k)_w = (f, \phi_k)_w

"""
from sympy import Symbol, cos
import numpy as np
import matplotlib.pyplot as plt
from shenfun.chebyshev.bases import ShenDirichletBasis
from shenfun.chebyshev.matrices import ADDmat

# Use sympy to compute a rhs, given an analytical solution
a = -0.5
b = 1.5
x = Symbol("x")
u = (1-x**2)**2*cos(np.pi*4*x)*(x-0.25)**2 + a*(1 + x)/2. + b*(1 - x)/2.
f = u.diff(x, 2)

# Size of discretization
N = 128

ST = ShenDirichletBasis(quad="GC", bc=(a, b))
points, weights = ST.points_and_weights(N, ST.quad)

# Gauss-Chebyshev quadrature to compute rhs
fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points

# Compute right hand side
f_hat = np.zeros_like(fj)
f_hat = ST.scalar_product(fj, f_hat)
f2 = f_hat.repeat(36).reshape((N, 6, 6))

# Solve Poisson equation
A = ADDmat(np.arange(N).astype(float), scale=1.0)
A.testfunction.bc = (a, b)
f_hat = A.solve(f_hat)

# Test 3D
f2 = A.solve(f2)
assert np.allclose(f2[:, 1, 1], f_hat)

uq = np.zeros_like(fj)
uq = ST.backward(f_hat, uq)

uj = np.array([u.subs(x, i) for i in points], dtype=np.float)
assert np.allclose(uj, uq)

plt.figure()
plt.plot(points, uj)
plt.title("U")
plt.figure()
plt.plot(points, uq - uj)
plt.title("Error")
plt.show()
