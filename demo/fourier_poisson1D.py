r"""
Solve Poisson equation on (-2\pi, 2\pi) with periodic bcs

.. math::

    \nabla^2 u = f, u(2\pi) = u(-2\pi)

Use Fourier basis and find u in V such that::

    (v, div(grad(u))) = (v, f)    for all v in V

V is the Fourier basis span{exp(1jkx)}_{k=-N/2}^{N/2-1}

Use the method of manufactured solutions, and choose a
solution that is either real or complex.

"""
from sympy import Symbol, cos, sin, lambdify
import numpy as np
from shenfun import inner, grad, TestFunction, TrialFunction, Basis, Function, \
    Array
import os
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Use sympy to compute a rhs, given an analytical solution
x = Symbol("x")
ue = cos(4*x) + 1j*sin(6*x)
#ue = cos(4*x)
fe = ue.diff(x, 2)

# Lambdify for faster evaluation
ul = lambdify(x, ue, 'numpy')
fl = lambdify(x, fe, 'numpy')

# Size of discretization
N = 40

dtype = {True: np.complex, False: np.float}[ue.has(1j)]
ST = Basis(N, dtype=dtype, plan=True, domain=(-2*np.pi, 2*np.pi))
u = TrialFunction(ST)
v = TestFunction(ST)

X = ST.mesh(N)

# Get f on quad points and exact solution
fj = Array(ST, buffer=fl(X))
uj = Array(ST, buffer=ul(X))

# Compute right hand side
f_hat = Function(ST)
f_hat = inner(v, fj, output_array=f_hat)

# Solve Poisson equation
A = inner(grad(v), grad(u))
u_hat = Function(ST)
u_hat = A.solve(-f_hat, u_hat)

uq = ST.backward(u_hat)
u_hat = ST.forward(uq, u_hat, fast_transform=False)
uq = ST.backward(u_hat, uq, fast_transform=False)

assert np.allclose(uj, uq)

point = np.array([0.1, 0.2])
p = ST.eval(point, u_hat)
assert np.allclose(p, ul(point))

if plt is not None and not 'pytest' in os.environ:
    plt.figure()
    plt.plot(X, uj)
    plt.title("U")
    plt.figure()
    plt.plot(X, uq - uj)
    plt.title("Error")
    plt.show()
