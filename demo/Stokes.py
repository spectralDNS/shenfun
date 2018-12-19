r"""Solve Stokes equations using a coupled formulation

The Stokes equations are in strong form

.. math::

    -\nabla^2 u - \nabla p &= f \\
    \nabla \cdot u &= h \\
    u(x, y=\pm 1) &= 0 \\
    u(x=2\pi, y) &= u(x=0, y) \\
    p(x=2\pi, y) &= p(x=0, y)

where :math:`f` and :math:`g` are given functions of space.
In addition we require :math:`\int p d\ = 0`, which is achieved by
fixing the coefficient :math:`\hat{p}_{0, 0} = 0`.

We use a tensorproductspace with Fourier expansions in the x-direction and
a composite Chebyshev or Legendre basis in the y-direction for ``u`` and
a regular (no boundary conditions) Chebyshev or Legendre basis for ``p``.

"""
import os
import sys
import numpy as np
import scipy.sparse as sp
from mpi4py import MPI
from sympy import symbols, exp, sin, cos, lambdify
from shenfun import *

comm = MPI.COMM_WORLD
x, y = symbols("x,y")

# Some right hand side (manufactured solution)
uex = sin(2*y)*(1-y**2)
uey = sin(2*x)*(1-y**2)
pe = -0.1*sin(2*x)
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

N = (32, 32)
family = sys.argv[-1] if len(sys.argv) == 2 else 'Legendre'
K0 = Basis(N[0], 'Fourier', dtype='d', domain=(0, 2*np.pi))
SD = Basis(N[1], family, bc=(0, 0))
ST = Basis(N[1], family)

TD = TensorProductSpace(comm, (K0, SD), axes=(1, 0))
TT = TensorProductSpace(comm, (K0, ST), axes=(1, 0))
VT = VectorTensorProductSpace(TD)
Q = MixedTensorProductSpace([VT, TT])
X = TD.local_mesh(True)

up = TrialFunction(Q)
vq = TestFunction(Q)

u, p = up
v, q = vq

if family.lower() == 'chebyshev':
    A00 = inner(v, -div(grad(u)))
    A01 = inner(v, -grad(p))
else:
    A00 = inner(grad(v), grad(u))
    A01 = inner(div(v), p)
A10 = inner(q, div(u))

# Get f and h on quad points
fh = Array(Q)
fh[0] = flx(*X)
fh[1] = fly(*X)
fh[2] = hl(*X)

fh_hat = Function(Q)
fh_hat[:2] = inner(v, fh[:2], output_array=fh_hat[:2])
fh_hat[2] = inner(q, fh[2], output_array=fh_hat[2])
# We now need to take care of the case with Fourier wavenumber = 0.
# Create submatrix for block (2, 2). This submatrix will only be enabled for
# Fourier wavenumber k=0.
A11 = inner(p, q)
A11.scale = K0.broadcast_to_ndims(np.zeros(N[1]))
if comm.Get_rank() == 0: # enable only for Fourier k=0
    A11.scale[0] = 1
A11.mats[1][0][:] = 0      # Zero the matrix diagonal (the only diagonal)
A11.mats[1][0][0] = 1      # fixes p_hat[0, 0]
A11.mats[1][0][-1] = 1     # fixes p_hat[0, -1]. Required to avoid singular matrix
if family.lower() == 'chebyshev':
    # Have to ident global row (N[1]-2)*2, but only for k=0. This is a bit tricky
    # Need to modify block (2, 1) as well as fixing the 1 on the diagonal
    a10 = inner(q, div(u))[1]   # This TPMatrix will be used for k=0
    a10.scale = K0.broadcast_to_ndims(np.zeros(N[1]))
    A10[1].scale = K0.broadcast_to_ndims(np.ones(N[1]))
    if comm.Get_rank() == 0:
        a10.scale[0] = 1     # enable for k=0
        A10[1].scale[0] = 0  # disable for k=0
    am = a10.pmat.diags().toarray()
    am[0] = 0
    a10.mats[1] = extract_diagonal_matrix(am)
    a10.pmat = a10.mats[1]
    A10.append(a10)
A11 = [A11]

# set p_hat[0, 0] = 0 and p_hat[0, -1] = 0
fh_hat[2, 0, 0] = 0
fh_hat[2, 0, -1] = 0

# Create block matrix
M = BlockMatrix(A00+A01+A10+A11)

# Solve problem
up_hat = M.solve(fh_hat)
up = up_hat.backward()

# Exact solution
ux = ulx(*X)
uy = uly(*X)
pe = pl(*X)

error = [comm.reduce(np.linalg.norm(ux-up[0])),
         comm.reduce(np.linalg.norm(uy-up[1])),
         comm.reduce(np.linalg.norm(pe-up[2]))]

if comm.Get_rank() == 0:
    print('Error    u          v          p')
    print('     %2.4e %2.4e %2.4e' %(error[0], error[1], error[2]))
    assert np.all(abs(np.array(error)) < 1e-8), error

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.contourf(X[0], X[1], up[2], 100)
    plt.figure()
    plt.quiver(X[0], X[1], up[0], up[1])
    plt.figure()
    plt.spy(M.diags((0, 0)).toarray()) # The matrix for Fourier given wavenumber
    plt.figure()
    plt.contourf(X[0], X[1], up[0], 100)
    #plt.show()
