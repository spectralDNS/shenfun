r"""
Solve Klein-Gordon equation on [-2pi, 2pi]**3 with periodic bcs

    u_tt = div(grad(u)) - u + u*|u|**2         (1)

Discretize in time by defining f = u_t and use mixed formulation

    f_t = div(grad(u)) - u + u*|u|**2         (1)
    u_t = f                                   (2)

with both u(x, y, z, t=0) and f(x, y, z, t=0) given.

Using the Fourier basis for all three spatial directions.

"""
from sympy import symbols, exp, lambdify
import numpy as np
import six
import matplotlib.pyplot as plt
from mpi4py import MPI
from time import time
from numbers import Number
import h5py
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun import *
from spectralDNS.utilities import Timer

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
TV = VectorTensorProductSpace([T, T, T])

Kp0 = C2CBasis(N[0], domain=(-2*np.pi, 2*np.pi), padding_factor=1.5)
Kp1 = C2CBasis(N[1], domain=(-2*np.pi, 2*np.pi), padding_factor=1.5)
Kp2 = R2CBasis(N[2], domain=(-2*np.pi, 2*np.pi), padding_factor=1.5)
Tp = TensorProductSpace(comm, (Kp0, Kp1, Kp2), slab=False, **{'planner_effort': 'FFTW_MEASURE'})

# Turn on padding by commenting out:
Tp = T

X = T.local_mesh(True)
fu = Array(TT, False)
f, u = fu[:]
up = Array(Tp, False)
K = np.array(T.local_wavenumbers(True, True, True))

dfu = Array(TT)
df, du = dfu[:]

fu_hat = Array(TT)
f_hat, u_hat = fu_hat[:]

gradu = Array(TV, False)

# initialize (f initialized to zero, so all set)
u[:] = ul(*X)
u_hat = T.forward(u, u_hat)

uh = TrialFunction(T)
vh = TestFunction(T)

A = inner(uh, vh)
L = -inner(grad(vh), grad(uh)) / A - gamma

# Coupled equations with no linear terms in their own variables,
# so place everything in NonlinearRHS
count = 0
def NonlinearRHS(fu, fu_hat, dfu_hat):
    global count, up
    count += 1
    dfu_hat.fill(0)
    f_hat, u_hat = fu_hat[:]
    df_hat, du_hat = dfu_hat[:]
    up = Tp.backward(u_hat, up)
    df_hat = Tp.forward(gamma*up**3, df_hat)
    df_hat += L*u_hat
    du_hat[:] = f_hat
    return dfu_hat

if rank == 0:
    plt.figure()
    image = plt.contourf(X[1][..., 0], X[0][..., 0], u[..., N[2]//2], 100)
    plt.draw()
    plt.pause(1e-4)

def update(fu, fu_hat, t, tstep, **params):
    global gradu

    timer()
    transformed = False
    if rank == 0 and tstep % params['plot_tstep'] == 0 and params['plot_tstep'] > 0:
        fu = TT.backward(fu_hat, fu)
        f, u = fu[:]
        image.ax.clear()
        image.ax.contourf(X[1][..., 0], X[0][..., 0], u[..., N[2]//2], 100)
        plt.pause(1e-6)
        transformed = True

    for ts, sl in six.iteritems(params['write_slice_tstep']):
        if tstep % ts == 0:
            if transformed is False:
                fu = TT.backward(fu_hat, fu)
                transformed = True
            for s in sl:
                params['file'].write_slice_tstep(tstep, s, fu)

    if tstep % params['write_tstep'] == 0:
        if transformed is False:
            fu = TT.backward(fu_hat, fu)
            transformed = True
        params['file'].write_tstep(tstep, fu)

    if tstep % params['Compute_energy'] == 0:
        if transformed is False:
            fu = TT.backward(fu_hat, fu)
        f, u = fu[:]
        f_hat, u_hat = fu_hat[:]
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

if __name__ == '__main__':
    file0 = HDF5Writer("KleinGordon{}.h5".format(N[0]), ['u', 'f'], TT)
    par = {'write_slice_tstep': {10: [[slice(None), slice(None), 16],
                                      [slice(None), 8, slice(None)]]},
           'write_tstep': 50,
           'Compute_energy': 100,
           'plot_tstep': -1,
           'file': file0}
    dt = 0.005
    end_time = 10.
    integrator = ETDRK4(TT, N=NonlinearRHS, update=update, **par)
    #integrator = RK4(TT, N=NonlinearRHS, update=update)
    integrator.setup(dt)
    t0 = time()
    fu_hat = integrator.solve(fu, fu_hat, dt, (0, end_time))

    file0.close()
    timer.final(MPI, rank, True)

    if rank == 0:
        generate_xdmf("KleinGordon{}.h5".format(N[0]))
