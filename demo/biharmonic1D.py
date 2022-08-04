r"""
Solve biharmonic equation in 1D

    u''''(x) = f(x),

Use Biharmonic basis with four boundary conditions.

"""
import os
import sympy as sp
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, FunctionSpace, Array, \
    Function, la

# Use sympy to compute a rhs, given an analytical solution
# Allow for a non-standard domain. Reference domain is (-1, 1)
a, b = -2, 2
domain = (a, b)
x = sp.Symbol("x", real=True)
ue = sp.sin(4*x)*sp.exp(-x/2)
fe = ue.diff(x, 4)
# Set the nonzero values of the 4 boundary conditions
bcs = f"""u({a})={ue.subs(x, a).n()} &&
          u'({a})={ue.diff(x, 1).subs(x, a).n()} &&
          u({b})={ue.subs(x, b).n()} &&
          u'({b})={ue.diff(x, 1).subs(x, b).n()}
          """

def main(N, family):
    SD = FunctionSpace(N, family=family, bc=bcs, domain=domain)
    X = SD.mesh()
    u = TrialFunction(SD)
    v = TestFunction(SD)

    # Compute right hand side of biharmonic equation
    f_hat = inner(v, fe)

    # Get left hand side of biharmonic equation
    matrices = inner(v, div(grad(div(grad(u)))))

    # Function to hold the solution
    u_hat = Function(SD)

    # Create linear algebra solver
    H = la.Solver(matrices)
    u_hat = H(f_hat, u_hat)
    uj = u_hat.backward()

    # Compare with analytical solution
    uq = Array(SD, buffer=ue)
    error = np.sqrt(inner(1, (uj-uq)**2))
    print(f"biharmonic1D {SD.family():14s} L2 error = {error:2.6e}")

    if 'pytest' not in os.environ:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(X, uq)

        plt.figure()
        plt.plot(X, uj)

        plt.figure()
        plt.plot(X, uq-uj)
        plt.title('Error')
        plt.show()

    else:
        assert error < 1e-6

if __name__ == '__main__':
    for family in ('legendre', 'chebyshev', 'chebyshevu', 'ultraspherical', 'jacobi'):
        main(24, family)
