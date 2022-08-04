r"""
This module is used to test all sorts of boundary conditions
for bases using orthogonal polynomials, like::

    - Chebyshev first and second kind
    - Legendre
    - Jacobi

We solve the Poisson equation in 2D with periodic boundaris in the x- and
z-directions and some mixture of Neumann/Dirichlet in the y-direction.

.. math::

    \nabla^2 u(x, y, z) = f(x, y, z), \quad (x, y, z) \in [0, 2\pi] \times [a, b] \times [0, 2\pi]

Use a Fourier basis for the periodic x- and z-directions. For the x-direction
choose either one of::

    0 : u(x, a, z), u(x, b, z)
    1 : u_y(x, a, z), u_y(x, b, z)
    2 : u(x, a, z), u_y(x, b, z)
    3 : u_y(x, a, z), u(x, b, z)
    4 : u(x, a, z), u_y(x, a, z)
    5 : u(x, b, z), u_y(x, b, z)

Option 1 requires a constraint since it is a pure Neumann problem.
The constraint is set by fixing the zeroth basis function such
that :math:`\int_a^b u w dx` is in agreement with the analytical
solution. The boundary condition along the edges may be functions
of the x- and z-coordinates. The exact value of the boundary condition is
computed from the manufactured solution. See `bcs` dictionary below.

"""
import os
import sympy as sp
import numpy as np
from shenfun import inner, div, grad, TestFunction, TrialFunction, \
    Array, Function, FunctionSpace, TensorProductSpace, la, \
    dx, comm
from mpi4py_fft.pencil import Subcomm

# Use sympy to compute a rhs, given an analytical solution
x, y, z = sp.symbols("x,y,z", real=True)
a = -2
b = 1
domain = (a, b)

ue = sp.cos(2*sp.pi*(y+0.1)/2)*(sp.sin(2*x)+sp.cos(3*x))*(sp.sin(3*z)+sp.cos(2*z))
fe = ue.diff(x, 2) + ue.diff(y, 2) + ue.diff(z, 2)

bcs = {
    0: {'left': {'D': ue.subs(y, a)}, 'right': {'D': ue.subs(y, b)}},
    1: {'left': {'N': ue.diff(y, 1).subs(y, a)}, 'right': {'N': ue.diff(y, 1).subs(y, b)}},
    2: {'left': {'D': ue.subs(y, a)}, 'right': {'N': ue.diff(y, 1).subs(y, b)}},
    3: {'left': {'N': ue.diff(y, 1).subs(y, a)}, 'right': {'D': ue.subs(y, b)}},
    4: {'left': {'D': ue.subs(y, a), 'N': ue.diff(y, 1).subs(y, a)}},
    5: {'right': {'D': ue.subs(y, b), 'N': ue.diff(y, 1).subs(y, b)}}
}

def main(N, family, bc):

    SD = FunctionSpace(N, family=family, bc=bcs[bc], domain=domain, alpha=1, beta=1) # alpha and beta are neglected for all but Jacobi
    K1 = FunctionSpace(N, family='F', dtype='D')
    K2 = FunctionSpace(N, family='F', dtype='d')

    # Try the uncommon approach of squeezing SD between the two Fourier spaces
    subcomms = Subcomm(comm, [0, 0, 1])
    T = TensorProductSpace(subcomms, (K1, SD, K2), axes=(1, 0, 2))
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
    #Solver = chebyshev.la.Helmholtz if family == 'chebyshev' and bc in (0, 1) else la.SolverGeneric1ND
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
        print(f'poisson3D L2 error = {error:2.6e}')
    if 'pytest 'in os.environ:
        assert error < 1e-8

if __name__ == '__main__':
    for family in ('legendre', 'chebyshev', 'chebyshevu', 'jacobi'):
        for bc in range(6):
            main(24, family, bc)
