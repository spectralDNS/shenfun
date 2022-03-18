# 2nd order rotational pressure correction for Navier-Stokes equation
# Author: Shashank Jaiswal, jaiswal0@purdue.edu

import numpy as np
import sympy as sp
from shenfun import *
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

# pylint: disable=multiple-statements

from mpltools import annotation
pa = {'fill': False, 'edgecolor': 'black'}
ta = {'fontsize': 10}

pex = lambda *args: print(*args) + exit(0)
x, y, t = sp.symbols("x, y, t", real=True)

# Define the initial solution
uex = (sp.sin(sp.pi*x)**2)*sp.sin(2*sp.pi*y)*sp.sin(t)
uey = -sp.sin(2*sp.pi*x)*(sp.sin(sp.pi*y)**2)*sp.sin(t)
pe = sp.cos(sp.pi*x)*sp.cos(sp.pi*y)*sp.sin(t)
fex = -uex.diff(x, 2) - uex.diff(y, 2) + pe.diff(x, 1) + uex.diff(t, 1) \
      + uex*uex.diff(x, 1) + uey*uex.diff(y, 1)
fey = -uey.diff(x, 2) - uey.diff(y, 2) + pe.diff(y, 1) + uey.diff(t, 1) \
      + uex*uey.diff(x, 1) + uey*uey.diff(y, 1)
he = uex.diff(x, 1) + uey.diff(y, 1)

uexf, ueyf, pef, fexf, feyf = map(lambda v: sp.lambdify((x, y, t), v),
                                  (uex, uey, pe, fex, fey))

def main(n):

    # number of modes in x and y direction
    N = (32, 32)

    # basis function for velocity components in x and y directions: P_{N}
    D0X = FunctionSpace(N[0], 'Legendre', quad='GL', dtype='d', bc=(0, 0))
    D0Y = FunctionSpace(N[1], 'Legendre', quad='GL', dtype='d', bc=(0, 0))

    # basis function for pressure: P_{N-2}
    PX = FunctionSpace(N[0], 'Legendre', quad='GL')
    PY = FunctionSpace(N[1], 'Legendre', quad='GL')
    PX.slice = lambda: slice(0, N[0]-2)
    PY.slice = lambda: slice(0, N[1]-2)

    # define a multi-dimensional tensor product basis
    Vs = TensorProductSpace(comm, (D0X, D0Y))
    Ps = TensorProductSpace(comm, (PX, PY), modify_spaces_inplace=True)

    # Create vector space for velocity
    Ws = VectorSpace([Vs, Vs])
    Cs = TensorSpace([Ws, Ws])  # cauchy stress tensor

    # Create test and trial spaces for velocity and pressure
    u = TrialFunction(Ws); v = TestFunction(Ws)
    p = TrialFunction(Ps); q = TestFunction(Ps)
    X = Vs.local_mesh(True)

    # Define the initial solution on quadrature points at t=0
    U = Array(Ws, buffer=(uex.subs(t, 0), uey.subs(t, 0)))
    P = Array(Ps)
    F = Array(Ws, buffer=(fex.subs(t, 0), fey.subs(t, 0)))
    U0 = U.copy()

    # Define the coefficient vector
    U_hat = Function(Ws); U_hat = Ws.forward(U, U_hat)
    P_hat = Function(Ps); P_hat = Ps.forward(P, P_hat)
    F_hat = Function(Ws); F_hat = Ws.forward(F, F_hat)

    # Initial time, time step, final time
    ti, dt, tf = 0., 5e-3/n, 5e-2
    nsteps = int(np.ceil((tf - ti)/dt))
    dt = (tf - ti)/nsteps
    X = Ws.local_mesh(True)

    # Define the implicit operator for BDF-2
    Lb1 = BlockMatrix(inner(v, u*(1.5/dt)) + inner(grad(v), grad(u)))
    Lb2 = BlockMatrix(inner(-grad(q), grad(p)))

    # Define the implicit operator for Euler
    Le1 = BlockMatrix(inner(v, u*(1./dt)) + inner(grad(v), grad(u)))
    Le2 = BlockMatrix(inner(-grad(q), grad(p)))

    # Define the implicit operator for updating
    Lu1 = BlockMatrix([inner(v, u)])
    Lu2 = BlockMatrix([inner(q, p)])

    # temporary storage
    rhsU, rhsP = Function(Ws), Function(Ps)
    U0_hat = Function(Ws); U0_hat = Ws.forward(U, U0_hat)
    Ut_hat = Function(Ws); Ut_hat = Ws.forward(U, Ut_hat)
    P0_hat = Function(Ps); P0_hat = Ps.forward(P, P0_hat)
    Phi_hat = Function(Ps); Phi_hat = Ps.forward(P, Phi_hat)

    # Create work arrays for nonlinear part
    UiUj = Array(Cs)
    UiUj_hat = Function(Cs)

    # integrate in time
    time = ti

    # storage
    rhsU, rhsP = rhsU, rhsP
    u_hat, p_hat = U_hat, P_hat
    u0_hat, p0_hat = U0_hat, P0_hat
    ut_hat, phi_hat = Ut_hat, Phi_hat

    # Euler time-step

    # evaluate the forcing function
    F[0] = fexf(X[0], X[1], time+dt)
    F[1] = feyf(X[0], X[1], time+dt)

    # Solve (9.102)
    rhsU.fill(0)
    rhsU += -inner(v, grad(p_hat))
    rhsU += inner(v, F)
    rhsU += inner(v, u_hat/dt)
    U = Ws.backward(U_hat, U)
    UiUj = outer(U, U, UiUj)
    UiUj_hat = UiUj.forward(UiUj_hat)
    rhsU += -inner(v, div(UiUj_hat))
    ut_hat = Le1.solve(rhsU, u=ut_hat)

    # Solve (9.107)
    rhsP.fill(0)
    rhsP += (1/dt)*inner(q, div(ut_hat))
    phi_hat = Le2.solve(rhsP, u=phi_hat, constraints=((0, 0, 0),))

    # Update for next time step
    u0_hat[:] = u_hat; p0_hat[:] = p_hat

    # Update (9.107)
    rhsU.fill(0)
    rhsU += inner(v, ut_hat) - inner(v, dt*grad(phi_hat))
    u_hat = Lu1.solve(rhsU, u=u_hat)

    # Update (9.105)
    rhsP.fill(0)
    rhsP += inner(q, phi_hat) + inner(q, p_hat) - inner(q, div(ut_hat))
    p_hat = Lu2.solve(rhsP, u=p_hat, constraints=((0, 0, 0),))

    time += dt

    # BDF time step
    for step in range(2, nsteps+1):

        # evaluate the forcing function
        F[0] = fexf(X[0], X[1], time+dt)
        F[1] = feyf(X[0], X[1], time+dt)

        # Solve (9.102)
        rhsU.fill(0)
        rhsU += -inner(v, grad(p_hat))
        rhsU += inner(v, F)
        rhsU += inner(v, u_hat*2/dt) - inner(v, u0_hat*0.5/dt)
        U = Ws.backward(U_hat, U)
        UiUj = outer(U, U, UiUj)
        UiUj_hat = UiUj.forward(UiUj_hat)
        rhsU += -2*inner(v, div(UiUj_hat))
        U0 = Ws.backward(U0_hat, U0)
        UiUj = outer(U0, U0, UiUj)
        UiUj_hat = UiUj.forward(UiUj_hat)
        rhsU += inner(v, div(UiUj_hat))
        ut_hat = Lb1.solve(rhsU, u=ut_hat)

        # Solve (9.107)
        rhsP.fill(0)
        rhsP += 1.5/dt*inner(q, div(ut_hat))
        phi_hat = Lb2.solve(rhsP, u=phi_hat, constraints=((0, 0, 0),))

        # update for next time step
        u0_hat[:] = u_hat; p0_hat[:] = p_hat

        # Update (9.107, 9.105)
        rhsU.fill(0)
        rhsU += inner(v, ut_hat) - inner(v, ((2.*dt/3))*grad(phi_hat))
        u_hat = Lu1.solve(rhsU, u=u_hat)

        rhsP.fill(0)
        rhsP += inner(q, phi_hat) + inner(q, p_hat) - inner(q, div(ut_hat))
        p_hat = Lu2.solve(rhsP, u=p_hat, constraints=((0, 0, 0),))

        # increment time
        time += dt

    # Transform the solution to physical space
    UP = [*U_hat.backward(U), P_hat.backward(P)]

    # compute error
    Ue = Array(Ws, buffer=(uex.subs(t, tf), uey.subs(t, tf)))
    Pe = Array(Ps, buffer=(pe.subs(t, tf)))
    UPe = [*Ue, Pe]
    l2_error = list(map(np.linalg.norm, [u-ue for u, ue in zip(UP, UPe)]))
    return l2_error


if __name__ == "__main__":
    from time import time
    t0 = time()
    N = 2**np.arange(0, 4)
    E = np.zeros((3, len(N)))

    for (j, n) in enumerate(N):
        E[:, j] = main(n)
    print('Time = ', time()-t0)
    fig = plt.figure(figsize=(5.69, 4.27))
    ax = plt.gca()
    marks = ('or', '-g', '-ob')
    vars = (r'$u_x$', r'$u_y$', r'$p$')
    for i in range(3):
        plt.loglog(N, E[i, :], marks[i], label=vars[i])
        slope, intercept = np.polyfit(np.log(N[-2:]), np.log(E[i, -2:]), 1)
        if i != 1:
            annotation.slope_marker((N[-2], E[i, -2]), ("{0:.2f}".format(slope), 1),
                                    ax=ax, poly_kwargs=pa, text_kwargs=ta)

    plt.text(N[0], 2e-5, r"$\Delta t=5 \times 10^{-3},\; N=32^2$")
    plt.text(N[0], 1e-5, r"Final Time = $5 \times 10^{-2}$")
    plt.title(r"Navier-Stokes: $2^{nd}$-order Rotational Pressure-Correction")
    plt.legend(); plt.autoscale()
    plt.ylabel(r'$|Error|_{L^2}$')
    plt.xticks(N)
    ax.get_xaxis().set_minor_formatter(NullFormatter())
    fmt = lambda v: r"$\Delta t/{0}$".format(v) if v != 1 else r"$\Delta t$"
    plt.gca().set_xticklabels(list(map(fmt, N)))
    #plt.savefig("navier-stokes.pdf", orientation='portrait')
    plt.show()
