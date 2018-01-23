r"""
Solve Ginzburg-Landau equation on (-50, 50)x(-50, 50) with periodic bcs

    u_t = div(grad(u)) + u - (1+1.5i)*u*|u|**2         (1)

Use Fourier basis V and find u in VxV such that

    (v, div(grad(u))) = (v, f)    for all v in VxV

The Fourier basis is span{exp(1jkx)}_{k=-N/2}^{N/2-1} and
VxV is a tensor product space.

"""
from sympy import symbols, exp, lambdify
import numpy as np
import pyfftw
import matplotlib.pyplot as plt
from mpi4py import MPI
import _pickle
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun import inner, div, grad, TestFunction, TrialFunction, Function, \
    TensorProductSpace, Array, ETDRK4, ETD, HDF5Writer

comm = MPI.COMM_WORLD

# Use sympy to set up initial condition
x, y = symbols("x,y")
#ue = (1j*x + y)*exp(-0.03*(x**2+y**2))
ue = (x + y)*exp(-0.03*(x**2+y**2))
ul = lambdify((x, y), ue, 'numpy')

try:
    # Look for wisdom stored using pyfftw.export_wisdom
    f = open('wisdom128.measure', 'rb')
    wisdom = _pickle.load(f)
    pyfftw.import_wisdom(wisdom)
except:
    pass

# Size of discretization
N = (128, 128)

K0 = C2CBasis(N[0], domain=(-50., 50.))
K1 = C2CBasis(N[1], domain=(-50., 50.))
T = TensorProductSpace(comm, (K0, K1), **{'planner_effort': 'FFTW_MEASURE'})

Kp0 = C2CBasis(N[0], domain=(-50., 50.), padding_factor=1.5)
Kp1 = C2CBasis(N[1], domain=(-50., 50.), padding_factor=1.5)
Tp = TensorProductSpace(comm, (Kp0, Kp1), **{'planner_effort': 'FFTW_PATIENT'})

u = TrialFunction(T)
v = TestFunction(T)

# Turn on padding by commenting:
#Tp = T

X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
U = Array(T, False)
Up = Array(Tp, False)
U_hat = Array(T)

#initialize
U[:] = ul(*X)
U_hat = T.forward(U, U_hat)

def LinearRHS():
    A = inner(u, v)
    L = inner(grad(v), -grad(u)) / A + 1
    return L

def NonlinearRHS(u, u_hat, rhs):
    global Up, Tp
    rhs.fill(0)
    Up = Tp.backward(u_hat, Up)
    rhs = Tp.forward(-(1+1.5j)*Up*abs(Up)**2, rhs)
    return rhs

plt.figure()
image = plt.contourf(X[0], X[1], U.real, 100)
plt.draw()
plt.pause(1e-6)
count = 0
def update(u, u_hat, t, tstep, **params):
    if tstep % params['plot_step'] == 0 and params['plot_step'] > 0:
        u = T.backward(u_hat, u)
        image.ax.clear()
        image.ax.contourf(X[0], X[1], u.real, 100)
        plt.pause(1e-6)
        count += 1
        #plt.savefig('Ginzburg_Landau_{}_{}.png'.format(N[0], count))
    if tstep % params['write_tstep'] == 0:
        u = T.backward(u_hat, u)
        params['file'].write_tstep(tstep, u.real)

if __name__ == '__main__':
    file0 = HDF5Writer("Ginzburg_Landau_{}.h5".format(N[0]), ['u'], T)
    par = {'plot_step': 100,
           'write_tstep': 50,
           'file': file0}
    t = 0.0
    dt = 0.001
    end_time = 0.1
    integrator = ETDRK4(T, L=LinearRHS, N=NonlinearRHS, update=update, **par)
    integrator.setup(dt)
    U_hat = integrator.solve(U, U_hat, dt, (0, end_time))
