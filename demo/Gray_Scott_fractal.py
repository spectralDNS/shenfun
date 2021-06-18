r"""
Solve Gray-Scott's equations on (-1, 1)x(-1, 1) with periodic bcs.

The equations to solve are

   u_t = -e1*(-div(grad(u)))**(alpha1/2) + b*(1-u) - u*v**2         (1)
   v_t = -e2*(-div(grad(v)))**(alpha2/2) - (b+kappa)*v + u*v**2             (2)

Using Fourier basis F and a vector tensor product space for u and v
The tensor product space is FF = F \otimes F, and the vector space is
W = FF \times FF.
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
from sympy import symbols
from sympy.functions import erf
import numpy as np
import matplotlib.pyplot as plt
from mpi4py_fft import generate_xdmf
from shenfun import inner, div, grad, TestFunction, TrialFunction, Function, \
    HDF5File, ETDRK4, TensorProductSpace, VectorSpace, FunctionSpace, Array, \
    comm

# Use sympy to set up initial condition
x, y = symbols("x,y", real=True)

# Initial conditions
a = 0.0001
u0 = 0.5*(1-((0.5*(erf((x-0.04)/a)+1) - 0.5*(erf((x+0.04)/a)+1))*(0.5*(erf((y-0.04)/a)+1) - 0.5*(erf((y+0.04)/a)+1))))+0.5
v0 = 0.25*(0.5*(erf((x-0.04)/a)+1) - 0.5*(erf((x+0.04)/a)+1))*(0.5*(erf((y-0.04)/a)+1) - 0.5*(erf((y+0.04)/a)+1))

# Size of discretization
N = (200, 200)

K0 = FunctionSpace(N[0], 'F', dtype='D', domain=(-1., 1.))
K1 = FunctionSpace(N[1], 'F', dtype='d', domain=(-1., 1.))
T = TensorProductSpace(comm, (K0, K1))
u = TrialFunction(T)
v = TestFunction(T)

# For nonlinear term we can use the 3/2-rule with padding
Tp = T.get_dealiased((1.5, 1.5))

# Turn on padding by commenting
#Tp = T

# Create vector spaces and a test function for the regular vector space
TV = VectorSpace(T)
TVp = VectorSpace(Tp)
vv = TestFunction(TV)
uu = TrialFunction(TV)

# Declare solution arrays and work arrays
UV = Array(TV, buffer=(u0, v0))
UVp = Array(TVp)
U, V = UV  # views into vector components
UV_hat = Function(TV)
w0 = Function(TV)         # Work array spectral space
w1 = Array(TVp)           # Work array physical space

e1 = 0.00002
e2 = 0.00001
b0 = 0.03

#initialize
UV_hat = UV.forward(UV_hat)

def LinearRHS(self, u, alpha1, alpha2, **params):
    L = inner(vv, (e1, e2)*div(grad(u)))
    L = np.array([-(-L[0].scale)**(alpha1/2),
                  -(-L[1].scale)**(alpha2/2)])
    return L

def NonlinearRHS(self, uv, uv_hat, rhs, kappa, **params):
    global b0, UVp, w0, w1, TVp
    rhs.fill(0)
    UVp = TVp.backward(uv_hat, UVp) # 3/2-rule dealiasing for nonlinear term
    w1[0] = b0*(1-UVp[0]) - UVp[0]*UVp[1]**2
    w1[1] = -(b0+kappa)*UVp[1] + UVp[0]*UVp[1]**2
    w0 = TVp.forward(w1, w0)
    rhs += w0
    return rhs

plt.figure()
X = T.local_mesh(True)
image = plt.contourf(X[0], X[1], U.real, 100)
plt.draw()
plt.pause(1)
uv0 = np.zeros_like(UV)
def update(self, uv, uv_hat, t, tstep, **params):
    if tstep % params['plot_step'] == 0 and params['plot_step'] > 0:
        uv = uv_hat.backward(uv)
        image.ax.clear()
        image.ax.contourf(X[0], X[1], uv[0].real, 100)
        plt.pause(1e-6)
        print(np.linalg.norm(uv[0]-uv0[0]),
              np.linalg.norm(uv[1]-uv0[1]),
              np.linalg.norm(uv[0]),
              np.linalg.norm(uv[1]))
        uv0[:] = uv

    if tstep % params['write_tstep'][0] == 0:
        uv = uv_hat.backward(uv)
        params['file'].write(tstep, params['write_tstep'][1], as_scalar=True)

if __name__ == '__main__':
    file0 = HDF5File("Gray_Scott_{}.h5".format(N[0]), mode='w')
    par = {'plot_step': 200,
           'write_tstep': (200, {'uv': [UV]}),
           'file': file0,
           'kappa': 0.061,
           'alpha1': 1.5,
           'alpha2': 1.9}
    dt = 10.
    end_time = 1000000
    integrator = ETDRK4(TV, L=LinearRHS, N=NonlinearRHS, update=update, **par)
    integrator.setup(dt)
    UV_hat = integrator.solve(UV, UV_hat, dt, (0, end_time))
    generate_xdmf("Gray_Scott_{}.h5".format(N[0]))
