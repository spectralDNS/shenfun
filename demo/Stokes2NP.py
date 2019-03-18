r"""Solve Stokes equations using a coupled formulation

The Stokes equations are in strong form

.. math::

    -\nabla^2 u - \nabla p &= f \\
    \nabla \cdot u &= h \\
    u(x, y=\pm 1) &= 0 \\
    u(x=\pm 1, y) &= 0

where :math:`f` and :math:`g` are given functions of space.
In addition we require :math:`\int p d\ = 0`, which is achieved by
fixing the coefficient :math:`\hat{p}_{0, 0} = 0`.

We use a tensorproductspace with a composite Legendre for the Dirichlet space
and a regular Legendre for the pressure space.

To remove all nullspaces we use a P_{N} x P_{N-2} basis, with P_{N-2} for the
pressure.

"""
import os
import numpy as np
from mpi4py import MPI
from sympy import symbols, sin, cos, lambdify
import scipy.sparse as sp
from shenfun import *

comm = MPI.COMM_WORLD
x, y = symbols("x,y")

assert comm.Get_size() == 1, "Two non-periodic directions only have solver implemented for serial"

# Some right hand side (manufactured solution)
uex = (cos(4*np.pi*x)+sin(2*np.pi*y))*(1-y**2)*(1-x**2)
uey = (sin(2*np.pi*x)+cos(6*np.pi*y))*(1-y**2)*(1-x**2)
pe = -0.1*sin(2*x)*sin(4*y)
fx = -uex.diff(x, 2) - uex.diff(y, 2) - pe.diff(x, 1)
fy = -uey.diff(x, 2) - uey.diff(y, 2) - pe.diff(y, 1)
h = uex.diff(x, 1) + uey.diff(y, 1)

# Lambdify for faster evaluation
ulx = lambdify((x, y), uex, 'numpy')
uly = lambdify((x, y), uey, 'numpy')
flx = lambdify((x, y), fx, 'numpy')
fly = lambdify((x, y), fy, 'numpy')
hl = lambdify((x, y), h, 'numpy')
pl = lambdify((x, y), pe, 'numpy')

N = (40, 40)
SD0 = Basis(N[0], 'Legendre', bc=(0, 0))
SD1 = Basis(N[1], 'Legendre', bc=(0, 0))
ST0 = Basis(N[0], 'Legendre')
ST1 = Basis(N[1], 'Legendre')

# To get a P_N x P_{N-2} space, just pick the first N-2 items of the pressure basis
# Note that this effectively sets P_N and P_{N-1} to zero, but still the basis uses
# the same quadrature points as the Dirichlet basis, which is required for the inner
# products.
ST0.slice = lambda: slice(0, ST0.N-2)
ST1.slice = lambda: slice(0, ST1.N-2)

TD = TensorProductSpace(comm, (SD0, SD1))
TT = TensorProductSpace(comm, (ST0, ST1))
VT = VectorTensorProductSpace(TD)
Q = MixedTensorProductSpace([VT, TT])
X = TD.local_mesh(True)

up = TrialFunction(Q)
vq = TestFunction(Q)

u, p = up
v, q = vq

# Assemble blocks of the complete block matrix
A00 = inner(grad(v), grad(u))
A01 = inner(div(v), p)
A10 = inner(q, div(u))

# Create submatrix for block (2, 2). This submatrix will only be used to fix pressure mode (0, 0).
A11 = inner(p, q)
for i in range(2):
    A11.mats[i][0][:] = 0      # Zero the matrix diagonal (the only diagonal)
    A11.mats[i][0][0] = 1      # fixes p_hat[0, 0]

# Create Block matrix
M = BlockMatrix(A00+A01+A10+[A11])

# Assemble right hand side
fh = Array(Q)
fh[0] = flx(*X)
fh[1] = fly(*X)
fh[2] = hl(*X)
fh_hat = Function(Q)
fh_hat[:2] = inner(v, fh[:2], output_array=fh_hat[:2])
fh_hat[2] = inner(q, fh[2], output_array=fh_hat[2])
fh_hat[2, 0, 0] = 0

# Solve problem
bh = np.zeros(Q.size(True))
bh[:] = fh_hat[:, :-2, :-2].ravel()
B = M.diags()
uh = sp.linalg.spsolve(B, bh)

# Move solution to regular Function
uh_hat = Function(Q)
uh_hat[:, :-2, :-2] = uh.reshape((3, N[0]-2, N[1]-2))
up = uh_hat.backward()

# Exact solution
ux = ulx(*X)
uy = uly(*X)
pe = pl(*X)

# Compute error
error = [comm.reduce(np.linalg.norm(ux-up[0])),
         comm.reduce(np.linalg.norm(uy-up[1])),
         comm.reduce(np.linalg.norm(pe-up[2]))]

if comm.Get_rank() == 0:
    print('Error    u          v          p')
    print('     %2.4e %2.4e %2.4e' %(error[0], error[1], error[2]))
    assert np.all(abs(np.array(error)) < 1e-7), error

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.contourf(X[0], X[1], up[2], 100)
    plt.figure()
    plt.quiver(X[0], X[1], up[0], up[1])
    plt.figure()
    plt.spy(B)
    plt.figure()
    plt.contourf(X[0], X[1], up[0], 100)
    #plt.show()
