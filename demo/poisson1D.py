r"""
Solve the Poisson equation in 1D

.. math::

    -\nabla^2 u(x) = f(x), \quad x \in [a, b]

where :math:`a < b`. The equation to solve is

.. math::

    -(\nabla^2 u, v) = (f, v)

and we need two boundary conditions. These boundary conditions
can be any combination of Dirichlet or Neumann, specified on
either side of the domain.

We create a function `main` that solves the problem by specifying
either one of::

    0 : u(a), u(b)
    1 : u'(a), u'(b)
    2 : u(a), u'(b)
    3 : u'(a), u(b)
    4 : u(a), u'(a)
    5 : u(b), u'(b)

Option 1 requires a constraint since it is a pure Neumann problem.
The constraint is set by fixing the zeroth basis function such
that :math:`\int_a^b u w dx` is in agreement with the analytical
solution.
"""
import os
import sympy as sp
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, FunctionSpace, \
    Array, Function, la, dx

# Use sympy to compute a rhs, given an analytical solution
x = sp.symbols("x", real=True)
ue = sp.cos(5*sp.pi*(x+0.1)/2)
fe = -ue.diff(x, 2)

a = -2
b = 2
domain = (a, b)

bcs = {
    0: f"u({a})={ue.subs(x, a).n()} && u({b})={ue.subs(x, b).n()}",
    1: f"u'({a})={ue.diff(x, 1).subs(x, a).n()} && u'({b})={ue.diff(x, 1).subs(x, b).n()}",
    2: f"u({a})={ue.subs(x, a).n()} && u'({b})={ue.diff(x, 1).subs(x, b).n()}",
    3: f"u'({a})={ue.diff(x, 1).subs(x, a).n()} && u({b})={ue.subs(x, b).n()}",
    4: f"u({a})={ue.subs(x, a).n()} && u'({a})={ue.diff(x, 1).subs(x, a).n()}",
    5: f"u({b})={ue.subs(x, b).n()} && u'({b})={ue.diff(x, 1).subs(x, b).n()}",
}

def main(N, family, bc):
    SD = FunctionSpace(N, family=family, domain=domain, bc=bcs[bc], alpha=0, beta=0) # alpha, beta are ignored by all other than jacobi
    u = TrialFunction(SD)
    v = TestFunction(SD)

    constraint = ()
    if bc == 1:
        # The Poisson equation with only Neumann boundary conditions require a constraint
        constraint = ((0, dx(Array(SD, buffer=ue), weighted=True)/dx(Array(SD, val=1), weighted=True)),)

    # Get f on quad points
    fj = Array(SD, buffer=fe)

    # Compute right hand side of Poisson equation
    f_hat = inner(v, fj)

    # Get left hand side of Poisson equation
    A0 = inner(v, -div(grad(u)))

    # Solve
    u_hat = Function(SD)
    M = la.Solver(A0)
    u_hat = M(f_hat, u_hat, constraints=constraint)

    # Transform to real space
    uj = u_hat.backward()

    # Compare with analytical solution
    ua = Array(SD, buffer=ue)
    print('L2 error = ', np.sqrt(inner(1, (uj-ua)**2)))
    if 'pytest 'in os.environ:
        assert np.sqrt(inner(1, (uj-ua)**2)) < 1e-5

if __name__ == '__main__':
    N = 36
    for family in ('legendre', 'chebyshev', 'chebyshevu', 'jacobi'):
        for bc in range(6):
            main(N, family, bc)
