r"""
Solve Klein-Gordon equation on [-2pi, 2pi]**3 with periodic bcs

    u_tt = div(grad(u)) - u + u*|u|**2         (1)

Discretize in time by defining f = u_t and use 4th order Runge-Kutta
to integrate forward in time

    f_t = div(grad(u)) - u + u*|u|**2         (1)
    u_t = f                                   (2)

with both u(x, y, z, t=0) and f(x, y, z, t=0) given.

Using the Fourier basis for all three spatial directions.

"""
from sympy import symbols, exp, lambdify
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from time import time
import h5py
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun import *
from spectralDNS.utilities import Timer
from shenfun.utilities.h5py_writer import HDF5Writer
from shenfun.utilities.generate_xdmf import generate_xdmf

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
timer = Timer()

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
T = TensorProductSpace(comm, (K0, K1, K2), slab=False, **{'planner_effort': 'FFTW_MEASURE'})

TT = MixedTensorProductSpace([T, T])

Kp0 = C2CBasis(N[0], domain=(-2*np.pi, 2*np.pi), padding_factor=1.5)
Kp1 = C2CBasis(N[1], domain=(-2*np.pi, 2*np.pi), padding_factor=1.5)
Kp2 = R2CBasis(N[2], domain=(-2*np.pi, 2*np.pi), padding_factor=1.5)
Tp = TensorProductSpace(comm, (Kp0, Kp1, Kp2), slab=False, **{'planner_effort': 'FFTW_MEASURE'})

# Turn on padding by commenting out:
Tp = T

file0 = HDF5Writer("KleinGordon{}.h5".format(N[0]), ['u', 'f'], TT)

X = T.local_mesh(True)
uf = Array(TT, False)
u, f = uf[:]
up = Array(Tp, False)

duf = Array(TT)
du, df = duf[:]

uf_hat = Array(TT)
uf_hat0 = Array(TT)
uf_hat1 = Array(TT)
w0 = Array(T)
u_hat, f_hat = uf_hat[:]

# initialize (f initialized to zero, so all set)
u[:] = ul(*X)
u_hat = T.forward(u, u_hat)

uh = TrialFunction(T)
vh = TestFunction(T)
A = inner(uh, vh)
k2 = -inner(grad(vh), grad(uh)) / A - gamma

count = 0
def compute_rhs(duf_hat, uf_hat, up, T, Tp, w0):
    global count
    count += 1
    duf_hat.fill(0)
    u_hat, f_hat = uf_hat[:]
    du_hat, df_hat = duf_hat[:]
    df_hat[:] = k2*u_hat
    up = Tp.backward(u_hat, up)
    df_hat += Tp.forward(gamma*up**3, w0)
    du_hat[:] = f_hat
    return duf_hat

# Integrate using a 4th order Rung-Kutta method
a = [1./6., 1./3., 1./3., 1./6.]         # Runge-Kutta parameter
b = [0.5, 0.5, 1.]                       # Runge-Kutta parameter
t = 0.0
dt = 0.005
end_time = 0.5
tstep = 0
write_x_slice = N[0]//2
#levels = np.linspace(-0.06, 0.1, 100)/8
#if rank == 0:
    #plt.figure()
    #image = plt.contourf(X[1][..., 0], X[0][..., 0], u[..., 16], 100)
    #plt.draw()
    #plt.pause(1e-4)
t0 = time()
K = np.array(T.local_wavenumbers(True, True, True))
TV = VectorTensorProductSpace([T, T, T])
gradu = Array(TV, False)
while t < end_time-1e-8:
    t += dt
    tstep += 1
    uf_hat1[:] = uf_hat0[:] = uf_hat
    for rk in range(4):
        duf = compute_rhs(duf, uf_hat, up, T, Tp, w0)
        if rk < 3:
            uf_hat[:] = uf_hat0 + b[rk]*dt*duf
        uf_hat1 += a[rk]*dt*duf
    uf_hat[:] = uf_hat1

    timer()

    if tstep % 10 == 0:
        uf = TT.backward(uf_hat, uf)
        file0.write_slice_tstep(tstep, [slice(None), slice(None), 16], uf)
        file0.write_slice_tstep(tstep, [slice(None), slice(None), 12], uf)

    if tstep % 25 == 0:
        uf = TT.backward(uf_hat, uf)
        file0.write_tstep(tstep, uf)

    if tstep % 100 == 0:
        uf = TT.backward(uf_hat, uf)
        ekin = 0.5*energy_fourier(f_hat, T)
        es = 0.5*energy_fourier(1j*K*u_hat, T)
        eg = gamma*np.sum(0.5*u**2 - 0.25*u**4)/np.prod(np.array(N))
        eg =  comm.allreduce(eg)
        gradu = TV.backward(1j*K*u_hat, gradu)
        ep = comm.allreduce(np.sum(f*gradu)/np.prod(np.array(N)))
        ea = comm.allreduce(np.sum(np.array(X)*(0.5*f**2 + 0.5*gradu**2 - (0.5*u**2 - 0.25*u**4)*f))/np.prod(np.array(N)))
        if rank == 0:
            print("Time = %2.2f Total energy = %2.8e Linear momentum %2.8e Angular momentum %2.8e" %(t, ekin+es+eg, ep, ea))
        comm.barrier()

file0.close()
timer.final(MPI, rank, True)

if rank == 0:
    generate_xdmf("KleinGordon{}.h5".format(N[0]))
