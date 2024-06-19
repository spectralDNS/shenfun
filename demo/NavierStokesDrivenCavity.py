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
import sympy as sp
from shenfun import *

assert comm.Get_size() == 1, "Two non-periodic directions only have solver implemented for serial"

Re = 250.
nu = 2./Re
alfa = 0.1 # underrelaxation factor
N = 64
family = 'Chebyshev'
#family = 'Legendre'
x = sp.symbols('x', real='True')
D0 = FunctionSpace(N, family, bc=(0, 0))
#D1 = FunctionSpace(N, family, bc=(0, 1))
D1 = FunctionSpace(N, family, bc=(0, (1-x)**2*(1+x)**2))

# Create tensor product spaces with different combination of bases
V1 = TensorProductSpace(comm, (D0, D1))
V0 = V1.get_homogeneous()
P = V1.get_orthogonal()

# To satisfy inf-sup for the Stokes problem, just pick the first N-2 items of the pressure basis
# Note that this effectively sets P_{N-1} and P_{N-2} to zero, but still the basis uses
# the same quadrature points as the Dirichlet basis, which is required for the inner products.
P.bases[0].slice = lambda: slice(0, N-2)
P.bases[1].slice = lambda: slice(0, N-2)

# Create vector space for velocity and a mixed velocity-pressure space
W1 = VectorSpace([V1, V0])
VQ = CompositeSpace([W1, P])

# Create space for nonlinearity
S1 = TensorSpace(P)

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

# Create Block matrix solver. This also takes care of boundary conditions.
sol = la.BlockMatrixSolver(A00+A01+A10)

# Create Functions to hold solution
up_hat = Function(VQ)
up_new = Function(VQ)

# Create regular work arrays for right hand side.
bh_hat = Function(VQ)

# Create arrays to hold velocity vector solution
ui = Array(W1)

# Create work arrays for nonlinear part
uiuj = Array(S1.get_dealiased())
uiuj_hat = Function(S1)
BS = BlockMatrix(inner(TestFunction(W1), div(TrialFunction(S1))))

def compute_rhs(up_hat, bh_hat):
    global uiuj, uiuj_hat
    bh_hat.fill(0)
    bi_hat = bh_hat[0]
    ui_hat = up_hat[0]
    # Get convection
    uip = ui_hat.backward(padding_factor=1.5)
    uiuj = outer(uip, uip, uiuj)
    uiuj_hat = uiuj.forward(uiuj_hat)
    #bi_hat = inner(v, div(uiuj_hat), output_array=bi_hat)
    bi_hat = BS.matvec(uiuj_hat, bi_hat) # fastest method
    #bi_hat = inner(grad(v), -uiuj_hat, output_array=bi_hat) # only Legendre
    #gradu = project(grad(ui_hat), S1)
    #bi_hat = inner(v, dot(gradu, ui_hat), output_array=bi_hat)
    return bh_hat

converged = False
count = 0
max_count = 1000
if 'pytest' in os.environ:
    max_count = 1
t0 = time.time()
up0 = up_hat[0].view()
up1 = up_new[0].view()
while not converged:
    count += 1
    bh_hat = compute_rhs(up_hat, bh_hat)
    up_new = sol(bh_hat, u=up_new, constraints=((2, 0, 0),)) # Constraint for component 2 of mixed space
    error = np.linalg.norm(up0-up1)
    up_hat.v[:] = alfa*up_new.v + (1-alfa)*up_hat.v
    converged = abs(error) < 1e-11 or count >= max_count
    if count % 1 == 0:
        print('Iteration %d Error %2.4e' %(count, error))

print('Time ', time.time()-t0)

# Move solution to regular Function
up = Array(VQ)
up = up_hat.backward(up)
u_, p_ = up

if 'pytest' in os.environ:
    sys.exit(1)

# Postprocessing
# Solve streamfunction
r = TestFunction(V0)
s = TrialFunction(V0)
S = inner(r, div(grad(s)))
h = inner(r, -curl(up_hat[0]))
H = la.SolverGeneric2ND(S)

phi_h = H(h)
phi = phi_h.backward()
# Compute vorticity
P1 = V1.get_orthogonal()
w_h = Function(P1)

w_h = project(curl(up_hat[0]), P1, output_array=w_h)
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
    converged = err < 1e-15 or count > 10
    psi_old = psi_min
    dx = dx/4.
    print("%d %d " %(xi, yi) +("%+2.7e "*4) %(xmid, ymid, psi_min, err))
    count += 1

# clean up
for T in [V0, V1, P, P1, uiuj.function_space(), up_hat._padded_space[1.5]]:
    T.destroy()
