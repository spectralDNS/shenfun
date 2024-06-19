r"""
This module is used to test all sorts of boundary conditions
for bases using orthogonal polynomials, like::

    - Chebyshev first and second kind
    - Legendre
    - Jacobi

We solve the Poisson equation in 2D with periodic boundaris in the y-direction
and some mixture of Neumann/Dirichlet in the x-direction

.. math::

    \nabla^2 u(x, y) = f(x, y), \quad (x, y) \in [a, b] \times [-2\pi, 2\pi]

Use a Fourier basis for the periodic y direction. For the x-direction choose
either one of::

    0 : u(a, y), u(b, y)
    1 : u_x(a, y), u_x(b, y)
    2 : u(a, y), u_x(b, y)
    3 : u_x(a, y), u(b, y)
    4 : u(a, y), u_x(a, y)
    5 : u(b, y), u_x(b, y)

Option 1 requires a constraint since it is a pure Neumann problem.
The constraint is set by fixing the zeroth basis function such
that :math:`\int_a^b u w dx` is in agreement with the analytical
solution. The boundary condition along the edges may be functions
of the y-coordinate. The exact value of the boundary condition is
computed from the manufactured solution. See `bcs` dictionary below.

"""
import os
import sympy as sp
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, \
    Array, Function, FunctionSpace, TensorProductSpace, comm, la, \
    chebyshev, dx

# Use sympy to compute a rhs, given an analytical solution
x, y = sp.symbols("x,y", real=True)
a = -1
b = 2
domain = (a, b)

ue = sp.cos(2*sp.pi*(x+0.1)/2)*(sp.sin(2*y)+sp.cos(3*y))
fe = ue.diff(x, 2) + ue.diff(y, 2)

bcs = {
    0: {'left': {'D': ue.subs(x, a)}, 'right': {'D': ue.subs(x, b)}},
    1: {'left': {'N': ue.diff(x, 1).subs(x, a)}, 'right': {'N': ue.diff(x, 1).subs(x, b)}},
    2: {'left': {'D': ue.subs(x, a)}, 'right': {'N': ue.diff(x, 1).subs(x, b)}},
    3: {'left': {'N': ue.diff(x, 1).subs(x, a)}, 'right': {'D': ue.subs(x, b)}},
    4: {'left': {'D': ue.subs(x, a), 'N': ue.diff(x, 1).subs(x, a)}},
    5: {'right': {'D': ue.subs(x, b), 'N': ue.diff(x, 1).subs(x, b)}}
}

def main(N, family, bc):
    SD = FunctionSpace(N, family=family, bc=bcs[bc], domain=domain, alpha=1, beta=1) # alpha and beta are neglected for all but Jacobi
    K1 = FunctionSpace(N, family='F', dtype='d', domain=(-2*sp.pi, 2*sp.pi))
    T = TensorProductSpace(comm, (SD, K1), axes=(0, 1))
    B = T.get_testspace(kind='PG')
    
    u = TrialFunction(T)
    v = TestFunction(B)

    constraint = ()
    if bc == 1:
        constraint = ((0, dx(Array(T, buffer=ue), weighted=True)/dx(Array(T, val=1), weighted=True)),)

    # Get f on quad points
    fj = Array(B, buffer=fe)

    # Compute right hand side of Poisson equation
    f_hat = Function(B)
    f_hat = inner(v, fj, output_array=f_hat)
    
    # Get left hand side of Poisson equation
    matrices = inner(v, div(grad(u)))
    
    # Create Helmholtz linear algebra solver
    #Solver = chebyshev.la.Helmholtz if family == 'C' and bc in (0, 1) else la.SolverGeneric1ND
    Solver = la.SolverGeneric1ND
    H = Solver(matrices)

    # Solve and transform to real space
    u_hat = Function(T)
    u_hat = H(f_hat, u_hat, constraints=constraint)
    uq = u_hat.backward()
    
    # Compare with analytical solution
    uj = Array(T, buffer=ue)
    
    error = np.sqrt(inner(1, (uj-uq)**2))
    if comm.Get_rank() == 0:
        print(f'poisson2D {family:s} L2 error = {error:2.6e}')
    if 'pytest 'in os.environ:
        assert error < 1e-6
    T.destroy()
    B.destroy()

if __name__ == '__main__':
    for family in 'CLUJ':
        for bc in range(6):
            main(28, family, bc)
