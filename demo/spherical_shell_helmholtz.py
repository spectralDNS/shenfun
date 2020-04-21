"""
Solve Helmholtz equation on a spherical shell

Using spherical coordinates

"""
from mpi4py import MPI
from shenfun import *
from shenfun.la import SolverGeneric1NP
import sympy as sp

by_parts = True

# Define spherical coordinates
r = 1
theta, phi = sp.symbols('x,y', real=True, positive=True)
psi = (theta, phi)
rv = (r*sp.sin(theta)*sp.cos(phi), r*sp.sin(theta)*sp.sin(phi), r*sp.cos(theta))

alpha = 2

# Manufactured solution
sph = sp.functions.special.spherical_harmonics.Ynm
ue = sph(10, 8, theta, phi)
#ue = sp.cos(8*(sp.sin(theta)*sp.cos(phi) + sp.sin(theta)*sp.sin(phi) + sp.cos(theta)))
g = - ue.diff(theta, 2) - (1/sp.tan(theta))*ue.diff(theta, 1) - (1/sp.sin(theta)**2)*ue.diff(phi, 2) + alpha*ue

N, M = 40, 30
# Choose domain for L0 somewhere in [0, pi], L1 somewhere in [0, 2pi]
L0 = Basis(N, 'L', bc=None, domain=(0, np.pi))
F1 = Basis(M, 'F', dtype='D')
T = TensorProductSpace(comm, (L0, F1), coordinates=(psi, rv))

v = TestFunction(T)
u = TrialFunction(T)

# Compute the right hand side on the quadrature mesh
gj = Array(T, buffer=g)

# Take scalar product
g_hat = Function(T)
g_hat = inner(v, gj, output_array=g_hat)

# Assemble matrices.
if by_parts:
    mats = inner(grad(v), grad(u))
    mats += [inner(v, alpha*u)]

else:
    mats = inner(v, -div(grad(u))+alpha*u)

# Solve
u_hat = Function(T)
Sol1 = SolverGeneric1NP(mats)
u_hat = Sol1(g_hat, u_hat)

# Transform back to real space.
uj = u_hat.backward()
uq = Array(T, buffer=ue)
print('Error =', np.linalg.norm(uj-uq))

# Postprocess
# Refine for a nicer plot. Refine simply pads Functions with zeros, which
# gives more quadrature points. u_hat has NxN quadrature points, refine
# using any higher number.
u_hat2 = u_hat.refine([N*2, M*2])
ur = u_hat2.backward()
# Get 2D array to plot on rank 0
from mayavi import mlab
xx, yy, zz = u_hat2.function_space().local_curvilinear_mesh()
# Wrap periodic direction around
if T.bases[1].domain == (0, 2*np.pi):
    xx = np.hstack([xx, xx[:, 0][:, None]])
    yy = np.hstack([yy, yy[:, 0][:, None]])
    zz = np.hstack([zz, zz[:, 0][:, None]])
    ur = np.hstack([ur, ur[:, 0][:, None]])
mlab.mesh(xx, yy, zz, scalars=ur.real, colormap='jet')
mlab.savefig('sphere.tiff')
mlab.show()

