r"""Solve Stoke's equations using a coupled formulation

The Stokes equations are in strong form

.. math::

    -\nabla^2 u - \nabla p &= f \\
    \nabla \cdot u &= h \\
    u(x, y=\pm 1) &= 0 \\
    u(x=2\pi, y) &= u(x=0, y)
    p(x=2\pi, y) &= p(x=0, y)

where :math:`f` and :math:`g` are functions of space.
In addition we require :math:`\int p d\ = 0`, which is achieved by
fixing :math:`\hat{p}_{0, 0} = 0`.

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
fex = -uex.diff(x, 2) - uex.diff(y, 2) - pe.diff(x, 1)
fey = -uey.diff(x, 2) - uey.diff(y, 2) - pe.diff(y, 1)
fp = uex.diff(x, 1) + uey.diff(y, 1)

# Lambdify for faster evaluation
ulx = lambdify((x, y), uex, 'numpy')
uly = lambdify((x, y), uey, 'numpy')
flx = lambdify((x, y), fex, 'numpy')
fly = lambdify((x, y), fey, 'numpy')
fpl = lambdify((x, y), fp, 'numpy')
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

# Get f and g on quad points
fvj = Array(Q)
fvj[0] = flx(*X)
fvj[1] = fly(*X)
fvj[2] = fpl(*X)

fv_hat = Function(Q)
fv_hat[:2] = inner(v, fvj[:2], output_array=fv_hat[:2])
fv_hat[2] = inner(q, fvj[2], output_array=fv_hat[2])
# We now need to take care of the case with Fourier wavenumber = 0.
# Create submatrix for block (2, 2). This submatrix will only be enabled for
# Fourier wavenumber k=0.
A11 = inner(p, q)
A11.scale = np.broadcast_to(A11.scale, (K0.shape(True), 1)).copy()
A11.scale[:] = 0
if comm.Get_rank() == 0: # enable only for Fourier k=0
    A11.scale[0] = 1
# Zero the matrix diagonal (the only diagonal)
A11.mats[1][0][:] = 0
A11.mats[1][0][0] = 1      # Fixes p_hat[0, 0]
A11.mats[1][0][-1] = 1     # fixes p_hat[0, -1]. Required to avoid singular matrix
A11 = [A11]

# set p_hat[0, 0] = 0 and p_hat[0, -1] = 0
fv_hat[2, 0, 0] = 0
fv_hat[2, 0, -1] = 0

# Create block matrix
M = BlockMatrix(A00+A01+A10+A11)

# Solve problem
up_hat = M.solve(fv_hat)
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
    plt.spy(M.diags((0, 0)).toarray()) # The matrix for Fourier wavenumber 0
    plt.figure()
    plt.contourf(X[0], X[1], up[0], 100)
    #plt.show()
