r"""
Solve Poisson equation on (0, 2pi) with periodic bcs

    \nabla^2 u = f, u(2pi) = u(0)

Use Fourier basis

"""
from sympy import Symbol, cos, sin, exp
import numpy as np
import matplotlib.pyplot as plt
from shenfun.fourier.bases import R2CBasis, C2CBasis
import shenfun
from shenfun import inner_product

# Use sympy to compute a rhs, given an analytical solution
x = Symbol("x")
u = cos(4*x)
#u = exp(1j*4*x)
f = u.diff(x, 2)

# Size of discretization
N = 32

ST = R2CBasis(N)

points = ST.points_and_weights()[0]

# Get f on quad points
fj = np.array([f.subs(x, j) for j in points], dtype=np.float)

# Compute right hand side
f_hat = inner_product(ST, fj, fast_transform=True)

#f2 = f_hat.repeat(36).reshape((N, 6, 6))
f2 = np.broadcast_to(f_hat[:, np.newaxis, np.newaxis], (N//2+1, 6, 6)).copy()
#f2 = np.broadcast_to(f_hat[:, np.newaxis, np.newaxis], (N, 6, 6)).copy()

# Solve Poisson equation
A = inner_product((ST, 0), (ST, 2))

f_hat = A.solve(f_hat)

# Test 3D
f2 = A.solve(f2)
assert np.allclose(f2[:, 1, 1], f_hat)

uq = np.zeros_like(fj)
uq = ST.backward(f_hat, uq, fast_transform=True)

uj = np.array([u.subs(x, i) for i in points], dtype=fj.dtype)
assert np.allclose(uj, uq)

plt.figure()
plt.plot(points, uj)
plt.title("U")
plt.figure()
plt.plot(points, uq - uj)
plt.title("Error")
plt.show()
