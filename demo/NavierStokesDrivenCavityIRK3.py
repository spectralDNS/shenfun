r"""Solve Navier-Stokes equations for the lid driven cavity using a coupled
formulation and a time-dependent implicit Runge-Kutta solver

The equations are in strong form

.. math::

    \frac{\partial u}{\partial t} + (u \cdot \nabla) u) &= \nu\nabla^2 u - \nabla p  \\
    \nabla \cdot u &= 0 \\
    \bs{u}(x, y=1) = (1, 0) \, &\text{ or }\, \bs{u}(x, y=1) = ((1-x)^2(1+x)^2, 0) \\
    u(x, y=-1) &= (0, 0) \\
    u(x=\pm 1, y) &= (0, 0)

In addition we require :math:`\int p \omega d\Omega = 0`, which is achieved by
fixing the coefficient :math:`\hat{p}_{0, 0} = 0`.

We use a tensorproductspace with a composite Legendre/Chebyshev for the Dirichlet space
and a regular Legendre/Chebyshev for the pressure space.

To remove all nullspaces (inf-sup) we use only N-2 basis functions for the pressure, and
N for each velocity component in each spatial direction.

"""
import functools
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from shenfun import *

x = sp.symbols('x', real='True')

assert comm.Get_size() == 1, "Two non-periodic directions have solver implemented only for serial"

class NavierStokesIRK3(IRK3):

    def __init__(self, N=(10, 10), dt=0.01, Re=100., modplot=10, family='C'):

        self.Re = Re
        self.nu = 2./Re
        self.N = N
        self.dt = dt
        self.modplot = modplot

        D0 = FunctionSpace(N[0], family, bc=(0, 0))
        #D1 = FunctionSpace(N[1], family, bc=(0, 1))
        D1 = FunctionSpace(N[1], family, bc=(0, (1-x)**2*(1+x)**2))

        # Create tensor product spaces with different combination of bases
        V1 = TensorProductSpace(comm, (D0, D1)) # velocity in x-direction
        V0 = V1.get_homogeneous()               # velocity in y-direction
        P = V1.get_orthogonal()                 # pressure

        # To satisfy inf-sup for the Stokes problem, just pick the first N-2 items of the pressure basis
        # Note that this effectively sets P_{N-1} and P_{N-2} to zero, but still the basis uses
        # the same quadrature points as the Dirichlet basis, which is required for the inner products.
        P.bases[0].slice = lambda: slice(0, N[0]-2)
        P.bases[1].slice = lambda: slice(0, N[1]-2)

        # Create vector space for velocity and a mixed velocity-pressure space
        W1 = VectorSpace([V1, V0])
        VQ = CompositeSpace([W1, P])

        # Functions to hold solution
        self.up_ = Array(VQ)
        self.up_hat = Function(VQ)

        # Create padded spaces for nonlinearity
        self.uip = Array(W1.get_dealiased())
        S1 = TensorSpace(V1.get_orthogonal().get_dealiased())
        self.uiuj = Array(S1)
        self.uiuj_hat = Function(S1)

        IRK3.__init__(self, VQ)

    def LinearRHS(self, up, *params):
        u, p = up
        return self.nu*div(grad(u))-grad(p)

    def NonlinearRHS(self, up, up_hat, rhs, **params):
        vq = TestFunction(self.T)
        v, q = vq
        rhs.fill(0)
        bi_hat = rhs[0]    # rhs vector for monentum equation
        ui_hat = up_hat[0] # velocity vector
        Wp = self.uip.function_space()
        uip = Wp.backward(ui_hat, self.uip)  # padded velocity vector
        uiuj = outer(uip, uip, self.uiuj)    # Reynolds stress
        uiuj_hat = uiuj.forward(self.uiuj_hat)
        bi_hat = inner(v, -div(uiuj_hat), output_array=bi_hat)
        return rhs

    def setup(self, dt):
        self.params['dt'] = dt
        up = TrialFunction(self.T)
        vq = TestFunction(self.T)
        u, p = up
        v, q = vq

        # Note that we are here assembling implicit left hand side matrices,
        # as well as matrices that can be used to assemble the right hande side
        # much faster through matrix-vector products
        a, b = self.a, self.b
        self.solver = []
        self.rhs_mats = []
        L = self.LinearRHS(up)
        A10 = inner(q, div(u))
        for rk in range(3):
            mats = inner(v, u - ((a[rk]+b[rk])*dt/2)*L)
            mats += A10
            sol = la.BlockMatrixSolver(mats)
            sol = functools.partial(sol, constraints=((2, 0, 0),)) # Constraint on pressure
            self.solver.append(sol)
            rhs_mats = inner(v, u + ((a[rk]+b[rk])*dt/2)*L)
            self.rhs_mats.append(BlockMatrix(rhs_mats))

    def update(self, up, up_hat, t, tstep, **par):
        if tstep % self.modplot == 0:
            up = up_hat.backward(up)
            u_ = up[0]
            plt.figure(1)
            X = self.T.local_mesh(True)
            plt.quiver(X[0], X[1], u_[0], u_[1])
            plt.pause(0.01)

if __name__ == '__main__':
    d = {'N': (32, 32),
         'dt': 0.1,
         'modplot': 100,
         'family': 'C',
         'Re': 250
        }
    sol = NavierStokesIRK3(**d)
    sol.solve(sol.up_, sol.up_hat, sol.dt, (0, 100))

    # Solve streamfunction
    r = TestFunction(sol.T[0][1])
    s = TrialFunction(sol.T[0][1])
    S = inner(r, div(grad(s)))
    h = inner(r, -curl(sol.up_hat[0]))
    H = la.SolverGeneric2ND(S)
    phi_h = H(h)
    phi = phi_h.backward()

    # Find minimal streamfunction value and position
    # by gradually zooming in on mesh
    W = 101
    converged = False
    xmid, ymid = 0, 0
    dx = 1
    psi_old = 0
    count = 0
    y, x = np.meshgrid(np.linspace(ymid-dx, ymid+dx, W), np.linspace(xmid-dx, xmid+dx, W))
    points = np.vstack((x.flatten(), y.flatten()))
    pp = phi_h.eval(points).reshape((W, W))
    while not converged:
        yr, xr = np.meshgrid(np.linspace(ymid-dx, ymid+dx, W), np.linspace(xmid-dx, xmid+dx, W))
        points = np.vstack((xr.flatten(), yr.flatten()))
        pr = phi_h.eval(points).reshape((W, W))
        xi, yi = pr.argmin()//W, pr.argmin()%W
        psi_min, xmid, ymid = pr.min()/2, xr[xi, yi], yr[xi, yi]
        err = abs(psi_min-psi_old)
        converged = err < 1e-12 or count > 10
        psi_old = psi_min
        dx = dx/4.
        print("%d %d " %(xi, yi) +("%+2.7e "*4) %(xmid, ymid, psi_min, err))
        count += 1
