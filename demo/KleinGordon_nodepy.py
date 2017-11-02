r"""
Solve Klein-Gordon equation on [-2pi, 2pi]**3 with periodic bcs

    u_tt = div(grad(u)) - u + u*|u|**2         (1)

Discretize in time by defining f = u_t and use 4th order Runge-Kutta
to integrate forward in time

    f_t = div(grad(u)) - u + u*|u|**2         (2)
    u_t = f                                   (3)

with both u(x, y, z, t=0) and f(x, y, z, t=0) given.

Using the Fourier basis for all three spatial directions.

"""
from sympy import symbols, exp, lambdify
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from time import time
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun import *
import nodepy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Use sympy to set up initial condition
x, y, z = symbols("x,y,z")
ue = 0.1*exp(-(x**2 + y**2 + z**2))
ul = lambdify((x, y, z), ue, 'numpy')

# Size of discretization
N = (32, 32, 32)

# Defocusing or focusing
gamma = 1

K0 = C2CBasis(N[0], domain=(-2*np.pi, 2*np.pi))
K1 = C2CBasis(N[1], domain=(-2*np.pi, 2*np.pi))
K2 = R2CBasis(N[2], domain=(-2*np.pi, 2*np.pi))
T = TensorProductSpace(comm, (K0, K1, K2), **{'planner_effort': 'FFTW_MEASURE'})

TT = MixedTensorProductSpace([T, T])
TV = VectorTensorProductSpace([T, T, T])

X = T.local_mesh(True)
fu = Array(TT, False) # solution real space
f, u = fu[:]          # split into views
up = Array(T, False)  # work array real space
w0 = Array(T)         # work array spectral space

dfu = Array(TT)       # solution spectral space
df, du = dfu[:]       # views

fu_hat = Array(TT)    # rhs array
f_hat, u_hat = fu_hat[:] # views

# initialize (f initialized to zero, so all set)
u[:] = ul(*X)
u_hat = T.forward(u, u_hat)

uh = TrialFunction(T)
vh = TestFunction(T)
A = inner(uh, vh)
k2 = -inner(grad(vh), grad(uh)) / A - gamma
count = 0

K = np.array(T.local_wavenumbers(True, True))
gradu = Array(TV, False)

if rank == 0:
    plt.figure()
    image = plt.contourf(X[1][..., 0], X[0][..., 0], u[..., N[2]//2], 100)
    plt.draw()
    plt.pause(1e-4)

def energy_fourier(comm, a):
    result = 2*np.sum(abs(a[..., 1:-1])**2) + np.sum(abs(a[..., 0])**2) + np.sum(abs(a[..., -1])**2)
    result =  comm.allreduce(result)
    return result

def update(t, fu_hat):
    """Callback to do some intermediate processing."""
    f_hat, u_hat = fu_hat[:]    # views
    fu[:] = TT.backward(fu_hat, fu)
    f, u = fu[:] # views
    ekin = 0.5*energy_fourier(T.comm, f_hat)
    es = 0.5*energy_fourier(T.comm, 1j*K*u_hat)
    eg = gamma*np.sum(0.5*u**2 - 0.25*u**4)/np.prod(np.array(N))
    eg =  comm.allreduce(eg)
    gradu[:] = TV.backward(1j*K*u_hat, gradu)
    ep = comm.allreduce(np.sum(f*gradu)/np.prod(np.array(N)))
    ea = comm.allreduce(np.sum(np.array(X)*(0.5*f**2 + 0.5*gradu**2 - (0.5*u**2 - 0.25*u**4)*f))/np.prod(np.array(N)))
    if rank == 0:
        image.ax.clear()
        image.ax.contourf(X[1][..., 0], X[0][..., 0], u[..., N[2]//2], 100)
        plt.pause(1e-6)
        #plt.savefig('Klein_Gordon_{}_real_{}.png'.format(N[0], tstep))
        print("Time = %2.2f Total energy = %2.8e Linear momentum %2.8e Angular momentum %2.8e" %(t, ekin+es+eg, ep, ea))

def rhs(t, fu):
    """Return right hand sides of Eq. (2) and (3)"""
    global up, count
    count += 1
    dfu.fill(0)
    f_hat, u_hat = fu[:]    # views
    df_hat, du_hat = dfu[:] # views
    df_hat[:] = k2*u_hat
    up = T.backward(u_hat, up)
    df_hat += T.forward(gamma*up**3, w0)
    du_hat[:] = f_hat
    if count % 400 == 0:
        update(t, fu)
    return dfu

# Integrate using nodepy
rk4 = nodepy.rk.loadRKM('RK44')
dt = 0.005
end_time = 10.
ivp = nodepy.ivp.IVP(f=rhs, T=[0., end_time], u0=fu_hat, name="KG")
t0 = time()
t, y = rk4(ivp, dt=dt, use_butcher=False)
print("Time ", time()-t0, count)

