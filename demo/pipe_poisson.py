"""
Solve Helmholtz equation in a pipe

Using cylindrical coordinates and numerical method from:

"Efficient spectral-Galerkin methods III: Polar and cylindrical geometries",
J Shen, SIAM J. Sci Comput. 18, 6, 1583-1604

Using shenfun to map coordinates instead of
directly applying r = (t+1)/2, as in the SIAM paper.

"""
import sympy as sp
import matplotlib.pyplot as plt
from shenfun import *
from shenfun.la import SolverGeneric1ND

by_parts = False

# Define polar coordinates using angle along first axis and radius second
r, theta, z = psi = sp.symbols('x,y,z', real=True, positive=True)
rv = (r*sp.cos(theta), r*sp.sin(theta), z)

alpha = 2

# Manufactured solution
ue = (r*(1-r)*sp.cos(4*theta)-1*(r-1))*sp.cos(4*z)
g = -ue.diff(r, 2) - (1/r)*ue.diff(r, 1) - (1/r**2)*ue.diff(theta, 2) - ue.diff(z, 2) + alpha*ue

N = 32
F0 = FunctionSpace(N, 'F', dtype='D')
F1 = FunctionSpace(N, 'F', dtype='d')
L = FunctionSpace(N, 'L', bc='Dirichlet', domain=(0, 1))
F2 = FunctionSpace(1, 'F', dtype='D')
F3 = FunctionSpace(N, 'F', dtype='d')
L0 = FunctionSpace(N, 'L', bc='UpperDirichlet', domain=(0, 1))
T = TensorProductSpace(comm, (L, F0, F1), coordinates=(psi, rv))
T0 = TensorProductSpace(MPI.COMM_SELF, (L0, F2, F3), coordinates=(psi, rv))

v = TestFunction(T)
u = TrialFunction(T)
v0 = TestFunction(T0)
u0 = TrialFunction(T0)

# Compute the right hand side on the quadrature mesh
gj = Array(T, buffer=g)

# Take scalar product
g_hat = Function(T)
g_hat = inner(v, gj, output_array=g_hat)
if T.local_slice(True)[1].start == 0:
    g_hat[:, 0] = 0 # Not using this basis for m=0, so this makes sure u_hat[:, 0] is zero

# For m=0 we solve only a 2D equation. Do the scalar product fo Fourier coefficient 0 by hand (or sympy)
if comm.Get_rank() == 0:
    g0_hat = Function(T0)
    X0 = T0.mesh()
    gt = sp.lambdify((r, theta, z), sp.integrate(g, (theta, 0, 2*sp.pi))/2/sp.pi)(*X0)
    g0_hat = T0.scalar_product(gt, g0_hat)

# Assemble matrices.
if by_parts:
    mats = inner(grad(v), grad(u))
    mats += [inner(v, alpha*u)]

    # case m=0
    if comm.Get_rank() == 0:
        mats0 = inner(grad(v0), grad(u0))
        mats0 += [inner(v0, alpha*u0)]

else:
    mats = inner(v, -div(grad(u))+alpha*u)

    # case m=0
    if comm.Get_rank() == 0:
        mats0 = inner(v0, -div(grad(u0))+alpha*u0)

# Solve
# case m > 0
u_hat = Function(T)
Sol1 = SolverGeneric1ND(mats)
u_hat = Sol1(g_hat, u_hat)

# case m = 0
u0_hat = Function(T0)
if comm.Get_rank() == 0:
    Sol0 = SolverGeneric1ND(mats0)
    u0_hat = Sol0(g0_hat, u0_hat)
comm.Bcast(u0_hat, root=0)

#K = F2.wavenumbers()
#for k in K[0]:
#    MM = (k**2+alpha)*C0 + M0
#    u0_hat[:-1, k] = MM.solve(g0_hat[:-1, k], u0_hat[:-1, k])

# Transform back to real space. Broadcast 1D solution
sl = T.local_slice(False)
uj = u_hat.backward() + u0_hat.backward()[sl[0], :, sl[2]]
ue = Array(T, buffer=ue)
print('Error =', np.linalg.norm(uj-ue))

# Postprocess
# Refine for a nicer plot. Refine simply pads Functions with zeros, which
# gives more quadrature points. u_hat has NxN quadrature points, refine
# using any higher number.
u_hat2 = u_hat.refine([N*2, N*2, N*2])
u0_hat2 = u0_hat.refine([N*2, 1, N*2])
sl = u_hat2.function_space().local_slice(False)
ur = u_hat2.backward() + u0_hat2.backward()[sl[0], :, sl[2]]
# Get 2D array to plot on rank 0
ur = ur.get((slice(None), slice(None), 2))
xx, yy, zz = u_hat2.function_space().curvilinear_mesh()

if comm.Get_rank() == 0:
    # Wrap periodic plot around since it looks nicer
    xp = np.hstack([xx[:, :, 0], xx[:, 0, 0][:, None]])
    yp = np.hstack([yy[:, :, 0], yy[:, 0, 0][:, None]])
    up = np.hstack([ur, ur[:, 0][:, None]])

    # plot
    plt.figure()
    plt.contourf(xp, yp, up)
    plt.colorbar()
    plt.show()
