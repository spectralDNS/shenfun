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
import os
from sympy import Symbol, cos, sin, lambdify
import numpy as np
from shenfun import inner, grad, TestFunction, TrialFunction, FunctionSpace, Function, \
    Array

# Use sympy to compute a rhs, given an analytical solution
x = Symbol("x", real=True)
ue = cos(4*x) + 1j*sin(6*x)
#ue = cos(4*x)
fe = ue.diff(x, 2)

# Size of discretization
N = 40

dtype = {True: np.complex, False: np.float}[ue.has(1j)]
ST = FunctionSpace(N, dtype=dtype, domain=(-2*np.pi, 2*np.pi))
u = TrialFunction(ST)
v = TestFunction(ST)

# Get f on quad points and exact solution
fj = Array(ST, buffer=fe)
uj = Array(ST, buffer=ue)

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
assert np.allclose(p, lambdify(x, ue)(point))

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    X = ST.mesh()
    plt.plot(X, uj.real)
    plt.title("U")
    plt.figure()
    plt.plot(X, (uq - uj).real)
    plt.title("Error")
    plt.show()
