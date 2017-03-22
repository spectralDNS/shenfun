r"""
Solve Poisson equation on (0, 2pi) with periodic bcs

    \nabla^2 u = f, u(2pi) = u(0)

Use Fourier basis

"""
from sympy import Symbol, cos, sin, exp
import numpy as np
import matplotlib.pyplot as plt
from shenfun.fourier.bases import R2CBasis
from shenfun import inner_product
from shenfun.operators import div, grad

# Use sympy to compute a rhs, given an analytical solution
x = Symbol("x")
u = cos(4*x)
f = u.diff(x, 2)

# Size of discretization
N = 32

ST = R2CBasis(N, plan=True)

points = ST.points_and_weights()[0]

# Get f on quad points
fj = np.array([f.subs(x, j) for j in points], dtype=np.float)

# Compute right hand side
f_hat = ST.scalar_product(fj)

# Solve Poisson equation
v = ST.test_function()
A = inner_product(v, div(grad(v)))
f_hat = A.solve(f_hat)

uq = ST.backward(f_hat)

uj = np.array([u.subs(x, i) for i in points], dtype=fj.dtype)
assert np.allclose(uj, uq)

plt.figure()
plt.plot(points, uj)
plt.title("U")
plt.figure()
plt.plot(points, uq - uj)
plt.title("Error")
plt.show()
