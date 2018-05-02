r"""
Solve Helmholtz equation on (0, 2pi)x(0, 2pi)x(0, 2pi) with periodic bcs

.. math::

    \nabla^2 u + u = f,

Use Fourier basis and find :math:`u` in :math:`V^3` such that

.. math::

    (v, \nabla^2 u + u) = (v, f), \quad \forall v \in V^3

where V is the Fourier basis :math:`span{exp(1jkx)}_{k=-N/2}^{N/2-1}` and
:math:`V^3` is a tensorproductspace.

"""
from sympy import symbols, cos, sin, lambdify
import matplotlib.pyplot as plt
from shenfun import inner, div, grad, TestFunction, TrialFunction, Basis, \
    TensorProductSpace
from mpi4py import MPI

comm = MPI.COMM_WORLD

# Use sympy to compute a rhs, given an analytical solution
x, y, z = symbols("x,y,z")
ue = cos(4*x) + sin(4*y) + sin(6*z)
fe = ue.diff(x, 2) + ue.diff(y, 2) + ue.diff(z, 2) + ue

ul = lambdify((x, y, z), ue, 'numpy')
fl = lambdify((x, y, z), fe, 'numpy')

# Size of discretization
N = 16

K0 = Basis(N, 'F', dtype='D')
K1 = Basis(N, 'F', dtype='D')
K2 = Basis(N, 'F', dtype='d')
T = TensorProductSpace(comm, (K0, K1, K2), slab=True)
X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = fl(*X)

# Compute right hand side
f_hat = inner(v, fj)

# Solve Poisson equation
A = inner(v, u+div(grad(u)))
f_hat = A.solve(f_hat)

uq = T.backward(f_hat, fast_transform=True)

uj = ul(*X)
print(abs(uj-uq).max())
#assert np.allclose(uj, uq)

plt.figure()
plt.contourf(X[0][:,:,0], X[1][:,:,0], uq[:, :, 0])
plt.colorbar()

plt.figure()
plt.contourf(X[0][:,:,0], X[1][:,:,0], uj[:, :, 0])
plt.colorbar()

plt.figure()
plt.contourf(X[0][:,:,0], X[1][:,:,0], uq[:, :, 0]-uj[:, :, 0])
plt.colorbar()
plt.title('Error')
#plt.show()

#from shenfun import VectorTensorProductSpace, curl, project

#Tk = VectorTensorProductSpace(T)
#v = TestFunction(Tk)
##u_ = Function(Tk, False)
##u_[:] = np.random.random(u_.shape)
##u_hat = Function(Tk)
##u_hat = Tk.forward(u_, u_hat)
##w_hat = inner(v, curl(u_), uh_hat=u_hat)

##u0 = u_[0]
##inner(v, u_)

#uq = T.as_function(uq)
#du_hat = Function(Tk)
#f_hat = T.as_function(f_hat)
#du_hat = project(grad(uq), Tk, output_array=du_hat, uh_hat=f_hat)
#du = Function(Tk, False)
#du = Tk.backward(du_hat, du)

#dux = ue.diff(x, 1)
#duxl = lambdify((x, y, z), dux, 'numpy')
#duxj = duxl(*X)
#duy = ue.diff(y, 1)
#duyl = lambdify((x, y, z), duy, 'numpy')
#duyj = duyl(*X)
#duz = ue.diff(z, 1)
#duzl = lambdify((x, y, z), duz, 'numpy')
#duzj = duzl(*X)


#plt.figure()
#plt.contourf(X[0][:,:,0], X[1][:,:,0], du[0, :, :, 0])
#plt.colorbar()

##plt.show()
#assert np.allclose(duxj, du[0])
#assert np.allclose(duyj, du[1])
#assert np.allclose(duzj, du[2])

