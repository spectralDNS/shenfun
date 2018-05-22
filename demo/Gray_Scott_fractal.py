r"""
Solve Gray-Scott's equations on (-1, 1)x(-1, 1) with periodic bcs.

The equations to solve are

   u_t = -e1*(-div(grad(u)))**(alpha1/2) + b*(1-u) - u*v**2         (1)
   v_t = -e2*(-div(grad(v)))**(alpha2/2) - (b+kappa)*v + u*v**2             (2)

Using Fourier basis F and a vector tensor product space for u and v
The tensor product space is FF = FxF, and the vector space is W = [FF, FF]
The constant diffusion coefficients are e1 and e2. Furthermore, b and
kappa are two model constants. The parameters alpha1 and alpha2 represent
coefficients for fractional derivatives on the Laplacian.

The variational problem reads: Find uv = (u, v) in W such that

    (qr, uv_t) = (qr, (e1, e2)*(-div(grad(uv)))**((alpha1, alpha2)/2)) \\
                 + b*(q, 1-u) -(b+kappa)*(r, v) - (q, u*v**2) + (r, u*v**2)

for all qr = (q, r) in W

Initial conditions are given as

    u(t=0) = 1 for abs(x) > 0.04 and 0.50 for abs(x) < 0.04
    v(t=0) = 0 for abs(x) > 0.04 and 0.25 for abs(x) < 0.04

and for stability they are approximated using error functions.

"""
from sympy import Symbol, lambdify
from sympy.functions import erf
import numpy as np
import matplotlib.pyplot as plt
from shenfun import inner, div, grad, TestFunction, TrialFunction, Function, HDF5Writer,\
    ETDRK4, ETD, RK4, TensorProductSpace, VectorTensorProductSpace, Basis, Array
import scipy
from mpi4py import MPI

comm = MPI.COMM_WORLD

# Use sympy to set up initial condition
x = Symbol("x")
y = Symbol("y")

# Initial conditions
a = 0.0001
u0 = 0.5*(1-((0.5*(erf((x-0.04)/a)+1) - 0.5*(erf((x+0.04)/a)+1))*(0.5*(erf((y-0.04)/a)+1) - 0.5*(erf((y+0.04)/a)+1))))+0.5
v0 = 0.25*(0.5*(erf((x-0.04)/a)+1) - 0.5*(erf((x+0.04)/a)+1))*(0.5*(erf((y-0.04)/a)+1) - 0.5*(erf((y+0.04)/a)+1))

ul = lambdify((x, y), u0, modules=['numpy', {'erf': scipy.special.erf}])
vl = lambdify((x, y), v0, modules=['numpy', {'erf': scipy.special.erf}])

# Size of discretization
N = (200, 200)

K0 = Basis(N[0], 'F', dtype='D', domain=(-1., 1.))
K1 = Basis(N[1], 'F', dtype='d', domain=(-1., 1.))
T = TensorProductSpace(comm, (K0, K1))
X = T.local_mesh(True)
u = TrialFunction(T)
v = TestFunction(T)

# For nonlinear term we can use the 3/2-rule with padding
Kp0 = Basis(N[0], 'F', dtype='D', domain=(-1., 1.), padding_factor=1.5)
Kp1 = Basis(N[1], 'F', dtype='d', domain=(-1., 1.), padding_factor=1.5)
Tp = TensorProductSpace(comm, (Kp0, Kp1))

# Turn on padding by commenting
Tp = T

# Create vector spaces and a test function for the regular vector space
TV = VectorTensorProductSpace(T)
TVp = VectorTensorProductSpace(Tp)
vv = TestFunction(TV)
uu = TrialFunction(TV)

# Declare solution arrays and work arrays
UV = Array(TV, False)
UVp = Array(TVp, False)
U, V = UV  # views into vector components
UV_hat = Function(TV)
w0 = Array(TV)              # Work array spectral space
w1 = Array(TVp, False)      # Work array physical space

e1 = 0.00002
e2 = 0.00001
b0 = 0.03

#initialize
U[:] = ul(*X)
V[:] = vl(*X)
UV_hat = TV.forward(UV, UV_hat)

def LinearRHS(alpha1, alpha2, **params):
    L = inner(vv, (e1, e2)*div(grad(uu)))
    L = np.array([-(-L[0])**(alpha1/2),
                  -(-L[1])**(alpha2/2)])
    return L

def NonlinearRHS(uv, uv_hat, rhs, kappa, **params):
    global b0, UVp, w0, w1, TVp
    rhs.fill(0)
    UVp = TVp.backward(uv_hat, UVp) # 3/2-rule dealiasing for nonlinear term
    w1[0] = b0*(1-UVp[0]) - UVp[0]*UVp[1]**2
    w1[1] = -(b0+kappa)*UVp[1] + UVp[0]*UVp[1]**2
    w0 = TVp.forward(w1, w0)
    rhs += w0
    return rhs

plt.figure()
image = plt.contourf(X[0], X[1], U.real, 100)
plt.draw()
plt.pause(1)
uv0 = np.zeros_like(UV)
def update(uv, uv_hat, t, tstep, **params):
    if tstep % params['plot_step'] == 0 and params['plot_step'] > 0:
        uv = TV.backward(uv_hat, uv)
        image.ax.clear()
        image.ax.contourf(X[0], X[1], uv[0].real, 100)
        plt.pause(1e-6)
        print(np.linalg.norm(uv[0]-uv0[0]),
              np.linalg.norm(uv[1]-uv0[1]),
              np.linalg.norm(uv[0]),
              np.linalg.norm(uv[1]))
        uv0[:] = uv


if __name__ == '__main__':
    file0 = HDF5Writer("Gray_Scott_{}.h5".format(N[0]), ['u', 'v'], T)
    par = {'plot_step': 200,
           'write_tstep': 200,
           'file': file0,
           'kappa': 0.061,
           'alpha1': 1.5,
           'alpha2': 1.9}
    dt = 10.
    end_time = 10000000
    integrator = ETDRK4(TV, L=LinearRHS, N=NonlinearRHS, update=update, **par)
    integrator.setup(dt)
    UV_hat = integrator.solve(UV, UV_hat, dt, (0, end_time))

