r"""Solve Stokes equations using a coupled formulation

The Stokes equations are in strong form

.. math::

    \nabla^2 u - \nabla p &= f \\
    \nabla \cdot u &= h \\
    u(x, y, z=\pm 1) &= 0

where :math:`f` and :math:`g` are given functions of space.
In addition we require :math:`\int p d\ = 0`, which is achieved by
fixing the coefficient :math:`\hat{p}_{0, 0, 0} = 0`.

We use a tensorproductspace with Fourier expansions in the x- and
y-directions and a composite Chebyshev or Legendre basis in the z-direction
for ``u`` and a regular (no boundary conditions) Chebyshev or Legendre basis
for ``p``.

To eliminate a nullspace we use a P_N basis for the velocity and a P_{N-2}
basis for the pressure.
"""
import os
import sys
import numpy as np
from mpi4py import MPI
from sympy import symbols, sin, cos, lambdify
from shenfun import *

comm = MPI.COMM_WORLD
x, y, z = symbols("x,y,z")

# Some right hand side (manufactured solution)
uex = sin(2*y)*(1-z**2)
uey = sin(2*x)*(1-z**2)
uez = sin(2*z)*(1-z**2)
pe = -0.1*sin(2*x)*cos(4*y)
fx = uex.diff(x, 2) + uex.diff(y, 2) + uex.diff(z, 2) - pe.diff(x, 1)
fy = uey.diff(x, 2) + uey.diff(y, 2) + uey.diff(z, 2) - pe.diff(y, 1)
fz = uez.diff(x, 2) + uez.diff(y, 2) + uez.diff(z, 2) - pe.diff(z, 1)
h = uex.diff(x, 1) + uey.diff(y, 1) + uez.diff(z, 1)

# Lambdify for faster evaluation
ulx = lambdify((x, y, z), uex, 'numpy')
uly = lambdify((x, y, z), uey, 'numpy')
ulz = lambdify((x, y, z), uez, 'numpy')
flx = lambdify((x, y, z), fx, 'numpy')
fly = lambdify((x, y, z), fy, 'numpy')
flz = lambdify((x, y, z), fz, 'numpy')
hl = lambdify((x, y, z), h, 'numpy')
pl = lambdify((x, y, z), pe, 'numpy')

N = (20, 20, 20)
family = sys.argv[-1] if len(sys.argv) == 2 else 'Legendre'
K0 = Basis(N[0], 'Fourier', dtype='D', domain=(0, 2*np.pi))
K1 = Basis(N[1], 'Fourier', dtype='d', domain=(0, 2*np.pi))
SD = Basis(N[2], family, bc=(0, 0))
ST = Basis(N[2], family)

# To get a P_N x P_{N-2} space, just pick the first N-2 items of the pressure basis
# Note that this effectively sets P_{N-1} and P_{N-2} to zero, but still the basis uses
# the same quadrature points as the Dirichlet basis, which is required for the inner
# products.
ST.slice = lambda: slice(0, ST.N-2)

TD = TensorProductSpace(comm, (K0, K1, SD), axes=(2, 0, 1))
TT = TensorProductSpace(comm, (K0, K1, ST), axes=(2, 0, 1))
VT = VectorTensorProductSpace(TD)
Q = MixedTensorProductSpace([VT, TT])
X = TD.local_mesh(True)

up = TrialFunction(Q)
vq = TestFunction(Q)

u, p = up
v, q = vq

# Assemble blocks of complete matrix
if family.lower() == 'chebyshev':
    A = inner(v, div(grad(u)))
    G = inner(v, -grad(p))
else:
    A = inner(grad(v), -grad(u))
    G = inner(div(v), p)
D = inner(q, div(u))

# We now need to take care of the case with Fourier wavenumber l=m=0.
# Create submatrix for block (3, 3). This submatrix will only be enabled for
# Fourier wavenumbers l=m=0.
P = inner(p, q)
s = TD.shape(True)
P.scale = np.zeros((s[0], s[1], 1))
ls = TD.local_slice(True)
if ls[0].start == 0 and ls[1].start == 0:   # enable only for Fourier l=m=0
    P.scale[0, 0] = 1
P.mats[2][0][:] = 0      # Zero the matrix diagonal (the only diagonal)
P.mats[2][0][0] = 1      # fixes p_hat[0, 0, 0]
if family.lower() == 'chebyshev':
    # Have to ident global row (N[0]-2)*(N[1]-2)*(N[2]-2), but only for l=m=0.
    # This is a bit tricky.
    # For Legendre this row is already zero. With Chebyshev we need to modify
    # block (3, 2) as well as fixing the 1 on the diagonal of (3, 3)
    a0 = inner(q, div(u))[2]   # This TPMatrix will be used for l=m=0
    a0.scale = np.zeros((s[0], s[1], 1))
    D[2].scale = np.ones((s[0], s[1], 1))
    if ls[0].start == 0 and ls[1].start == 0:
        a0.scale[0, 0] = 1     # enable for l=m=0
        D[2].scale[0, 0] = 0  # disable for l=m=0
    am = a0.pmat.diags().toarray()
    am[0] = 0
    a0.mats[2] = extract_diagonal_matrix(am)
    a0.pmat = a0.mats[2]
    D.append(a0)
P = [P]

# Create block matrix
M = BlockMatrix(A+G+D+P)

# Get f and h on quad points
fh = Array(Q)
fh[0] = flx(*X)
fh[1] = fly(*X)
fh[2] = flz(*X)
fh[3] = hl(*X)

fh_hat = Function(Q)
fh_hat[:3] = inner(v, fh[:3], output_array=fh_hat[:3])
fh_hat[3] = inner(q, fh[3], output_array=fh_hat[3])
fh_hat[3, 0, 0, 0] = 0 # set p_hat[0, 0, 0] = 0

# Solve problem
up_hat = M.solve(fh_hat)
up = up_hat.backward()

# Exact solution
ux = ulx(*X)
uy = uly(*X)
uz = ulz(*X)
pe = pl(*X)

error = [comm.reduce(np.linalg.norm(ux-up[0])),
         comm.reduce(np.linalg.norm(uy-up[1])),
         comm.reduce(np.linalg.norm(uz-up[2])),
         comm.reduce(np.linalg.norm(pe-up[3]))]

if comm.Get_rank() == 0:
    print('Error    u          v          w        p')
    print('     %2.4e %2.4e %2.4e %2.4e' %(error[0], error[1], error[2], error[3]))
    assert np.all(abs(np.array(error)) < 1e-8), error

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.contourf(X[0][:, :, 0], X[1][:, :, 0], up[2, :, :, 6], 100)
    plt.figure()
    plt.quiver(X[0][:, :, 0], X[1][:, :, 0], up[0, :, :, 6], up[1, :, :, 6])
    plt.figure()
    l, m = 5, 5
    plt.spy(M.diags((l, m, 0)), markersize=2, color='k') # The matrix for Fourier given wavenumber
    plt.title('Block matrix: l, m = ({}, {})'.format(l, m))
    plt.xticks([])
    plt.yticks([])
    plt.figure()
    plt.contourf(X[0][:, :, 0], X[1][:, :, 0], up[0, :, :, 6], 100)
    plt.show()
