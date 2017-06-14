r"""
Solve Poisson equation in 3D with periodic bcs in two directions
and homogeneous Dirichlet in the third

    \nabla^2 u = f,

Use Fourier basis for the periodic directions and Shen's Dirichlet basis for the
remaining non-periodic direction. Discretization leads to a Holmholtz problem.

Note that the equation to solve for Legendre basis is

     (\nabla u, \nabla v) = -(f, v)

whereas for Chebyshev we solve

     (\nabla^2 u, v) = (f, v)

"""
import sys
import importlib
from sympy import symbols, cos, sin, exp, lambdify
import numpy as np
import matplotlib.pyplot as plt
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun.tensorproductspace import TensorProductSpace
from shenfun import inner, div, grad, TestFunction, TrialFunction, Function, \
    project, Dx
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD

# Collect basis and solver from either Chebyshev or Legendre submodules
basis = sys.argv[-1] if len(sys.argv) == 2 else 'chebyshev'
shen = importlib.import_module('.'.join(('shenfun', basis)))
Basis = shen.bases.ShenDirichletBasis
Solver = shen.la.Helmholtz

# Use sympy to compute a rhs, given an analytical solution
a = -1
b = 1
x, y, z = symbols("x,y,z")
ue = (cos(4*x) + sin(2*y) + sin(4*z))*(1-y**2) + a*(1 + y)/2. + b*(1 - y)/2.
fe = ue.diff(x, 2) + ue.diff(y, 2) + ue.diff(z, 2)

# Lambdify for faster evaluation
ul = lambdify((x, y, z), ue, 'numpy')
fl = lambdify((x, y, z), fe, 'numpy')

# Size of discretization
N = (14, 15, 16)

SD = Basis(N[0], bc=(a, b))
K1 = C2CBasis(N[1])
K2 = R2CBasis(N[2])
T = TensorProductSpace(comm, (K1, SD, K2), axes=(1,0,2))
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
u = TrialFunction(T)
v = TestFunction(T)

K = T.local_wavenumbers()

# Get f on quad points
fj = fl(*X)

# Compute right hand side of Poisson equation
f_hat = inner(v, fj)
if basis == 'legendre':
    f_hat *= -1.

# Get left hand side of Poisson equation
if basis == 'chebyshev':
    matrices = inner(v, div(grad(u)))
else:
    matrices = inner(grad(v), grad(u))

# Create Helmholtz linear algebra solver
H = Solver(**matrices)

# Solve and transform to real space
u_hat = Function(T)           # Solution spectral space
t0 = time.time()
u_hat = H(u_hat, f_hat)       # Solve
uq = T.backward(u_hat, fast_transform=False)

# Compare with analytical solution
uj = ul(*X)
print(abs(uj-uq).max())
assert np.allclose(uj, uq)

plt.figure()
plt.contourf(X[0][:,:,0], X[1][:,:,0], uq[:, :, 2])
plt.colorbar()

plt.figure()
plt.contourf(X[0][:,:,0], X[1][:,:,0], uj[:, :, 2])
plt.colorbar()

plt.figure()
plt.contourf(X[0][:,:,0], X[1][:,:,0], uq[:, :, 2]-uj[:, :, 2])
plt.colorbar()
plt.title('Error')

#plt.show()

from shenfun import VectorTensorProductSpace, curl, project, Expr
B = shen.bases.Basis(N[0])
TT = TensorProductSpace(comm, (K1, B, K2), axes=(1,0,2))

Tk = VectorTensorProductSpace([TT]*3)
v = TestFunction(Tk)
#u_ = Function(Tk, False)
#u_[:] = np.random.random(u_.shape)
#u_hat = Function(Tk)
#u_hat = Tk.forward(u_, u_hat)
#w_hat = inner(v, curl(u_), uh_hat=u_hat)

#u0 = u_[0]
#inner(v, u_)

#uq = T.as_function(uq)
uu = Function(TT, False)
uu[:] = uq
du_hat = Function(Tk)
u_hat = T.as_function(u_hat)
du_hat = project(grad(uu), Tk, output_array=du_hat)
du = Function(Tk, False)
du = Tk.backward(du_hat, du)

dux = ue.diff(x, 1)
duxl = lambdify((x, y, z), dux, 'numpy')
duxj = duxl(*X)
duy = ue.diff(y, 1)
duyl = lambdify((x, y, z), duy, 'numpy')
duyj = duyl(*X)
duz = ue.diff(z, 1)
duzl = lambdify((x, y, z), duz, 'numpy')
duzj = duzl(*X)


plt.figure()
plt.contourf(X[0][:,:,0], X[1][:,:,0], du[0, :, :, 0])
plt.colorbar()

#plt.show()
assert np.allclose(duxj, du[0])
assert np.allclose(duyj, du[1])
assert np.allclose(duzj, du[2])

vq = Function(Tk, False)
vq[:] = np.random.random(vq.shape)
vq_hat = Function(Tk)
vq_hat = Tk.forward(vq, vq_hat)
dv_hat = Function(Tk)
dv_hat = project(Expr(3*vq), Tk, output_array=dv_hat, uh_hat=vq_hat)

#p = T.forward.output_pencil
#print([c.Get_size() for c in p.subcomm])
