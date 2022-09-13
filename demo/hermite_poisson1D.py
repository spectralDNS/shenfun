r"""
Solve Poisson equation in 1D with homogeneous Dirichlet bcs on the domain (-inf, inf)

.. math::

    \nabla^2 u = f,

The equation to solve for a Hermite basis is

.. math::

     (\nabla u, \nabla v) = -(f, v)

"""
import os
from sympy import symbols, exp, hermite, lambdify
import numpy as np
from shenfun import inner, grad, TestFunction, TrialFunction, \
    Array, Function, FunctionSpace

# Use sympy to compute a rhs, given an analytical solution
x = symbols("x", real=True)
#ue = sin(4*x)*exp(-x**2)
ue = hermite(4, x)*exp(-x**2/2)
fe = ue.diff(x, 2)

def main(N):
    SD = FunctionSpace(N, 'Hermite')
    u = TrialFunction(SD)
    v = TestFunction(SD)

    # Get f on quad points
    fj = Array(SD, buffer=fe)

    # Compute right hand side of Poisson equation
    f_hat = Function(SD)
    f_hat = inner(v, -fj, output_array=f_hat)

    # Get left hand side of Poisson equation
    A = inner(grad(v), grad(u))

    f_hat = A / f_hat
    uj = f_hat.backward()
    uh = uj.forward()

    # Compare with analytical solution
    ua = Array(SD, buffer=ue)
    error = np.sqrt(inner(1, (uj-ua)**2))
    print(f"hermite_poisson1D L2 error {error:2.6e}")

    if 'pytest' not in os.environ:
        import matplotlib.pyplot as plt
        xx = np.linspace(-8, 8, 100)
        plt.plot(xx, lambdify(x, ue)(xx), 'r', xx, uh.eval(xx), 'bo', markersize=2)
        plt.show()

    else:
        assert error < 1e-6
        point = np.array([0.1, 0.2])
        p = SD.eval(point, f_hat)
        assert np.allclose(p, lambdify(x, ue)(point), atol=1e-5)

if __name__ == '__main__':
    main(10)
