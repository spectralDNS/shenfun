"""
Solve Biharmonic equation on the unit disc

Using polar coordinates and numerical method from:

"Efficient spectral-Galerkin methods III: Polar and cylindrical geometries",
J Shen, SIAM J. Sci Comput. 18, 6, 1583-1604

"""
import os
from shenfun import *
from shenfun.la import SolverGeneric1ND
import sympy as sp

# Define polar coordinates using angle along first axis and radius second
theta, r = psi = sp.symbols('x,y', real=True, positive=True)
rv = (r*sp.cos(theta), r*sp.sin(theta))

N = 36
by_parts = False
F = FunctionSpace(N, 'F', dtype='d')
F0 = FunctionSpace(1, 'F', dtype='d')
L = FunctionSpace(N, 'L', bc='Bipolar', domain=(0, 1))
L0 = FunctionSpace(N, 'L', bc='BiPolar0', domain=(0, 1))
T = TensorProductSpace(comm, (F, L), axes=(1, 0), coordinates=(psi, rv))
T0 = TensorProductSpace(MPI.COMM_SELF, (F0, L0), axes=(1, 0), coordinates=(psi, rv))

v = TestFunction(T)
u = TrialFunction(T)
v0 = TestFunction(T0)
u0 = TrialFunction(T0)

# Manufactured solution
ue = (r*(1-r))**4*(1+sp.cos(8*theta))

# Right hand side of numerical solution
g = div(grad(div(grad(u)))).tosympy(basis=ue, psi=psi)

#g = (ue.diff(r, 4)
#     + (2/r**2)*ue.diff(r, 2, theta, 2)
#     + 1/r**4*ue.diff(theta, 4)
#     + (2/r)*ue.diff(r, 3)
#     - 2/r**3*ue.diff(r, 1, theta, 2)
#     - 1/r**2*ue.diff(r, 2)
#     + 4/r**4*ue.diff(theta, 2)
#     + 1/r**3*ue.diff(r, 1))

# Compute the right hand side on the quadrature mesh
gj = Array(T, buffer=g)

# Take scalar product
g_hat = Function(T)
g_hat = inner(v, gj, output_array=g_hat)
if T.local_slice(True)[0].start == 0: # The processor that owns k=0
    g_hat[0] = 0

# For m=0 we solve only a 1D equation. Do the scalar product for Fourier coefficient 0 by hand (or sympy)
if comm.Get_rank() == 0:
    g0_hat = Function(T0)
    gt = sp.lambdify(r, sp.integrate(g, (theta, 0, 2*sp.pi))/2/sp.pi)(L0.mesh())
    g0_hat = T0.scalar_product(gt, g0_hat)

if by_parts:
    mats = inner(div(grad(v)), div(grad(u)))
    if comm.Get_rank() == 0:
        mats0 = inner(div(grad(v0)), div(grad(u0)))

else:
    mats = inner(v, div(grad(div(grad(u)))))
    if comm.Get_rank() == 0:
        mats0 = inner(v0, div(grad(div(grad(u0)))))

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

# Transform back to real space.
sl = T.local_slice(False)
uj = u_hat.backward() + u0_hat.backward()[:, sl[1]]
ue = Array(T, buffer=ue)
X = T.local_mesh(True)
print('Error =', np.linalg.norm(uj-ue))
assert np.linalg.norm(uj-ue) < 1e-8

if 'pytest' not in os.environ:
    import matplotlib.pyplot as plt

    # Postprocess
    # Refine for a nicer plot. Refine simply pads Functions with zeros, which
    # gives more quadrature points. u_hat has NxN quadrature points, refine
    # using any higher number.
    u_hat2 = u_hat.refine([N*3, N*3])
    u0_hat2 = u0_hat.refine([1, N*3])
    sl = u_hat2.function_space().local_slice(False)
    ur = u_hat2.backward() + u0_hat2.backward()[:, sl[1]]

    # Wrap periodic plot around since it looks nicer
    xx, yy = u_hat2.function_space().local_cartesian_mesh()
    xp = np.vstack([xx, xx[0]])
    yp = np.vstack([yy, yy[0]])
    up = np.vstack([ur, ur[0]])

    # plot
    plt.figure()
    plt.contourf(xp, yp, up)
    plt.colorbar()
    plt.title('Biharmonic - unitdisc')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()
