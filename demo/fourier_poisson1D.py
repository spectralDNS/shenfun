r"""
Solve Poisson equation on (0, 2pi) with periodic bcs

    \nabla^2 u = f, u(2pi) = u(0)

Use Fourier basis and find u in V such that

    (v, div(grad(u))) = (v, f)    for all v in V

V is the Fourier basis span{exp(1jkx)}_{k=-N/2}^{N/2-1}

Use the method of manufactured solutions, and choose a
solution that is either real or complex.

"""
from sympy import Symbol, cos, sin, lambdify
import numpy as np
from shenfun import inner, grad, TestFunction, TrialFunction
from shenfun.fourier.bases import FourierBasis
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
N = 32

dtype = {True: np.complex, False: np.float}[ue.has(1j)]
ST = FourierBasis(N, dtype, plan=True, domain=(-np.pi, np.pi))
u = TrialFunction(ST)
v = TestFunction(ST)

X = ST.mesh(N)

# Get f on quad points and exact solution
fj = fl(X)
uj = ul(X)

# Compute right hand side
f_hat = inner(v, fj)

# Solve Poisson equation
A = inner(grad(v), grad(u))
u_hat = A.solve(-f_hat)

uq = ST.backward(u_hat)

assert np.allclose(uj, uq)

if not plt is None and not 'pytest' in os.environ:
    plt.figure()
    plt.plot(X, uj)
    plt.title("U")
    plt.figure()
    plt.plot(X, uq - uj)
    plt.title("Error")
    plt.show()
