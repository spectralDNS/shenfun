r"""Solve Navier-Stokes equations for the lid driven cavity using a coupled
formulation

The equations are in strong form

.. math::

    \nu\nabla^2 u - \nabla p &= (u \cdot \nabla) u) \\
    \nabla \cdot u &= 0 \\
    i\bs{u}(x, y=1) = (1, 0) \, &\text{ or }\, \bs{u}(x, y=1) = ((1-x)^2(1+x)^2, 0) \\
    u(x, y=-1) &= 0 \\
    u(x=\pm 1, y) &= 0

In addition we require :math:`\int p d\ = 0`, which is achieved by
fixing the coefficient :math:`\hat{p}_{0, 0} = 0`.

We use a tensorproductspace with a composite Legendre for the Dirichlet space
and a regular Legendre for the pressure space.

To remove all nullspaces we use a P_{N} x P_{N-2} basis, with P_{N-2} for the
pressure.

"""
import os
import sys
import time
import numpy as np
from mpi4py import MPI
from scipy.sparse.linalg import splu
import sympy
from shenfun import *

assert comm.Get_size() == 1, "Two non-periodic directions only have solver implemented for serial"

Re = 10.
nu = 2./Re
alfa = 0.1 # underrelaxation factor
N = (45, 45)
family = 'Chebyshev'
#family = 'Legendre'
quad = 'GC'
x = sympy.symbols('x', real='True')
D0X = Basis(N[0], family, quad=quad, bc=(0, 0))
#D1Y = Basis(N[1], family, quad=quad, bc=(1, 0))
D1Y = Basis(N[1], family, quad=quad, bc=((1-x)**2*(1+x)**2, 0))
D0Y = Basis(N[1], family, quad=quad, bc=(0, 0))
PX = Basis(N[0], family, quad=quad)
PY = Basis(N[1], family, quad=quad)

# To get a P_N x P_{N-2} space, just pick the first N-2 items of the pressure basis
# Note that this effectively sets P_N and P_{N-1} to zero, but still the basis uses
# the same quadrature points as the Dirichlet basis, which is required for the inner
# products.
PX.slice = lambda: slice(0, PX.N-2)
PY.slice = lambda: slice(0, PY.N-2)

# Create tensor product spaces with different combination of bases
V1 = TensorProductSpace(comm, (D0X, D1Y))
V0 = TensorProductSpace(comm, (D0X, D0Y))
P = TensorProductSpace(comm, (PX, PY))

# Create vector space for velocity
W1 = VectorTensorProductSpace([V1, V0])

# Create mixed space for total solution
VQ = MixedTensorProductSpace([W1, P])   # for velocity and pressure

# Create padded spaces for nonlinearity
V1p = V1.get_dealiased((1.5, 1.5))
V0p = V0.get_dealiased((1.5, 1.5))
#V1p = V1.get_dealiased(dealias_direct=True)
#V0p = V0.get_dealiased(dealias_direct=True)
#V1p = V1 # Or do not use dealiasing at all. Makes very little difference here
#V0p = V0
W1p = VectorTensorProductSpace([V1p, V0p])
W0p = VectorTensorProductSpace([V0p, V0p])
QTp = MixedTensorProductSpace([W1p, W0p])  # for uiuj

up = TrialFunction(VQ)
vq = TestFunction(VQ)

u, p = up
v, q = vq

# Assemble blocks of the complete block matrix
if family.lower() == 'legendre':
    A00 = inner(grad(v), -nu*grad(u))
    A01 = inner(div(v), p)
else:
    A00 = inner(v, nu*div(grad(u)))
    A01 = inner(v, -grad(p))

A10 = inner(q, div(u))

# Extract the boundary matrices
bc_mats = extract_bc_matrices([A00, A01, A10])

# Create Block matrix
M = BlockMatrix(A00+A01+A10)

# Create Function to hold solution
uh_hat = Function(VQ)
ui_hat = uh_hat[0]
D1Y.bc.set_boundary_dofs(ui_hat[0], True)

# New solution (iterative)
uh_new = Function(VQ)
ui_new = uh_new[0]
D1Y.bc.set_boundary_dofs(ui_new[0], True)

# Compute the constant contribution to rhs due to nonhomogeneous boundary conditions
bh_hat0 = Function(VQ)
BM = BlockMatrix(bc_mats)
bh_hat0 = BM.matvec(-uh_hat, bh_hat0)
bi_hat0 = bh_hat0[0]

# Create regular work arrays for right hand side. (Note that bc part will not be used so we can use Q)
bh_hat = Function(VQ)

# Create arrays to hold velocity vector solution
ui = Array(W1)
uip = Array(W1p)

# Create work arrays for nonlinear part
uiuj = Array(QTp)
uiuj_hat = Function(QTp)

def compute_rhs(ui_hat, bh_hat):
    global uip, uiuj, uiuj_hat, W1p
    bh_hat.fill(0)
    uip = W1p.backward(ui_hat, uip)
    uiuj = outer(uip, uip, uiuj)
    uiuj_hat = uiuj.forward(uiuj_hat)
    bi_hat = bh_hat[0]
    bi_hat = inner(v, div(uiuj_hat), output_array=bi_hat)
    #bi_hat = inner(grad(v), -uiuj_hat, output_array=bi_hat)
    bh_hat += bh_hat0
    return bh_hat

uh_hat, Ai = M.solve(bh_hat0, u=uh_hat, constraints=((2, 0, 0),), return_system=True) # Constraint for component 2 of mixed space
Alu = splu(Ai)
uh_new[:] = uh_hat
converged = False
count = 0
max_count = 1000
if 'pytest' in os.environ:
    max_count = 1
t0 = time.time()
while not converged:
    count += 1
    bh_hat = compute_rhs(ui_hat, bh_hat)
    uh_new = M.solve(bh_hat, u=uh_new, constraints=((2, 0, 0),), Alu=Alu) # Constraint for component 2 of mixed space
    error = np.linalg.norm(ui_hat-ui_new)
    uh_hat[:] = alfa*uh_new + (1-alfa)*uh_hat
    converged = abs(error) < 1e-11 or count >= max_count
    if count % 1 == 0:
        print('Iteration %d Error %2.4e' %(count, error))

print('Time ', time.time()-t0)

# Move solution to regular Function
up = uh_hat.backward()
u_, p_ = up

if 'pytest' in os.environ: sys.exit(0)

# Postprocessing
# Solve streamfunction
r = TestFunction(V0)
s = TrialFunction(V0)
S = inner(r, div(grad(s)))
h = inner(r, -curl(ui_hat))
H = la.SolverGeneric2NP(S)
phi_h = H(h)
phi = phi_h.backward()
# Compute vorticity
PX.slice = lambda: slice(0, PX.N)
PY.slice = lambda: slice(0, PY.N)
w_h = Function(P)
w_h = project(curl(ui_hat), P, output_array=w_h)
#p0 = np.array([[0.], [0.]])
#print(w_h.eval(p0)*2)

# Find minimal streamfunction value and position
# by gradually zooming in on mesh
W = 101
converged = False
xmid, ymid = 0, 0
dx = 1
psi_old = 0
count = 0
y, x = np.meshgrid(np.linspace(ymid-dx, ymid+dx, W), np.linspace(xmid-dx, xmid+dx, W))
points = np.vstack((x.flatten(), y.flatten()))
pp = phi_h.eval(points).reshape((W, W))
while not converged:
    yr, xr = np.meshgrid(np.linspace(ymid-dx, ymid+dx, W), np.linspace(xmid-dx, xmid+dx, W))
    points = np.vstack((xr.flatten(), yr.flatten()))
    pr = phi_h.eval(points).reshape((W, W))
    xi, yi = pr.argmin()//W, pr.argmin()%W
    psi_min, xmid, ymid = pr.min()/2, xr[xi, yi], yr[xi, yi]
    err = abs(psi_min-psi_old)
    converged = err < 1e-12 or count > 10
    psi_old = psi_min
    dx = dx/4.
    print("%d %d " %(xi, yi) +("%+2.7e "*4) %(xmid, ymid, psi_min, err))
    count += 1

import matplotlib.pyplot as plt
#f = open('plot_u_y_Ghia{}.csv'.format(int(Re)))
#g = np.loadtxt(f, skiprows=1, delimiter=',')
#plt.figure()
#y = 2*(g[:, 0]-0.5)
#plt.plot(y, g[:, 1], 'r+')
X = V0.local_mesh(True)
#x = np.vstack([np.zeros(N[0]), X[1][0]])
#res = ui_hat[0].eval(x)
#plt.plot(x[1], res)
#res2 = ui_hat[0].eval(np.vstack([np.zeros(len(y)), y]))
#plt.plot(y, res2, 'bs', mfc='None')
plt.figure()
plt.contourf(X[0], X[1], p_, 100)
plt.figure()
plt.quiver(X[0], X[1], u_[0], u_[1])
plt.figure()
plt.spy(M.diags())
plt.figure()
plt.contourf(X[0], X[1], u_[0], 100)
plt.figure()
plt.contourf(X[0], X[1], u_[1], 100)
plt.figure()
plt.contour(x, y, pp, 100)
plt.title('Streamfunction')
#plt.show()
