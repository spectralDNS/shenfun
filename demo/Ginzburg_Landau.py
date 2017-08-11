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
    TensorProductSpace, Array

comm = MPI.COMM_WORLD

# Use sympy to set up initial condition
x, y = symbols("x,y")
#ue = (1j*x + y)*exp(-0.03*(x**2+y**2))
ue = (x + y)*exp(-0.03*(x**2+y**2))
ul = lambdify((x, y), ue, 'numpy')

f = open('wisdom128.measure', 'rb')
wisdom = _pickle.load(f)
pyfftw.import_wisdom(wisdom)

# Size of discretization
N = (128, 128)

K0 = C2CBasis(N[0], domain=(-50., 50.))
K1 = C2CBasis(N[1], domain=(-50., 50.))
T = TensorProductSpace(comm, (K0, K1), **{'planner_effort': 'FFTW_MEASURE'})

Kp0 = C2CBasis(N[0], domain=(-50., 50.), padding_factor=1.6)
Kp1 = C2CBasis(N[1], domain=(-50., 50.), padding_factor=1.6)
Tp = TensorProductSpace(comm, (Kp0, Kp1), **{'planner_effort': 'FFTW_MEASURE'})

u = TrialFunction(T)
v = TestFunction(T)

# Turn on padding by commenting:
#Tp = T

X = T.local_mesh(True) # With broadcasting=True the shape of X is local_shape, even though the number of datapoints are still the same as in 1D
U = Array(T, False)
Up = Array(Tp, False)
dU = Array(T)
U_hat = Array(T)
U_hat0 = Array(T)
U_hat1 = Array(T)
w0 = Array(T)
a = [1./6., 1./3., 1./3., 1./6.]         # Runge-Kutta parameter
b = [0.5, 0.5, 1.]                       # Runge-Kutta parameter

A = (2*np.pi)**2  # Equals inner(u, v)

#initialize
U[:] = ul(*X)
U_hat = T.forward(U, U_hat)

k2 = inner(grad(v), -grad(u)).diagonal_array/A

#@profile
def compute_rhs(rhs, u_hat, U, Up, T, Tp, w0):
    rhs.fill(0)
    rhs = k2*u_hat
    rhs += u_hat
    Up = Tp.backward(u_hat, Up)
    rhs -= Tp.forward((1+1.5j)*Up*abs(Up)**2, w0)
    return rhs

# Integrate using a 4th order Rung-Kutta method
t = 0.0
dt = 0.05
end_time = 96.
tstep = 0
plt.figure()
image = plt.contourf(X[0], X[1], U.real, 100)
plt.draw()
plt.pause(1e-6)
while t < end_time-1e-8:
    t += dt
    tstep += 1
    U_hat1[:] = U_hat0[:] = U_hat
    for rk in range(4):
        dU = compute_rhs(dU, U_hat, U, Up, T, Tp, w0)
        if rk < 3:
            U_hat[:] = U_hat0 + b[rk]*dt*dU
        U_hat1 += a[rk]*dt*dU
    U_hat[:] = U_hat1

    if tstep % 100 == 0:
        U = T.backward(U_hat, U)
        image.ax.clear()
        image.ax.contourf(X[0], X[1], U.real, 100)
        plt.pause(1e-6)
        #plt.savefig('Ginzburg_Landau_pad_{}_real_{}.png'.format(N[0], int(np.round(t))))


U = T.backward(U_hat, U)

plt.figure()
plt.contourf(X[0], X[1], U.real, 100)
plt.colorbar()

plt.figure()
plt.contourf(X[0], X[1], U.imag, 100)
plt.colorbar()

plt.show()

