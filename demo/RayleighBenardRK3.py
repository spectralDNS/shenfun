from shenfun import *
import matplotlib.pyplot as plt
import sympy
np.warnings.filterwarnings('ignore')

# pylint: disable=attribute-defined-outside-init

x, y, tt = sympy.symbols('x,y,t', real=True)

comm = MPI.COMM_WORLD

class RayleighBenard:

    def __init__(self, N=(32, 32), domain=((-1, 1), (0, 2*np.pi)), Ra=10000., Pr=0.7, dt=0.1,
                 bcT=(0, 1), conv=0, modplot=100, modsave=1e8, filename='RB',
                 family='C', padding_factor=(1, 1.5), checkpoint=1000):
        self.nu = np.sqrt(Pr/Ra)
        self.kappa = 1./np.sqrt(Pr*Ra)
        self.dt = dt
        self.conv = conv
        self.modplot = modplot
        self.modsave = modsave
        self.bcT = bcT
        self.filename = filename

        self.a = (8./15., 5./12., 3./4.)
        self.b = (0.0, -17./60., -5./12.)
        self.c = (0.0, 8./15., 2./3., 1)

        # Regular spaces
        self.B0 = FunctionSpace(N[0], family, bc=(0, 0, 0, 0), domain=domain[0])
        self.D0 = FunctionSpace(N[0], family, bc=(0, 0), domain=domain[0])
        self.C0 = FunctionSpace(N[0], family, domain=domain[0])
        self.T0 = FunctionSpace(N[0], family, bc=bcT, domain=domain[0])
        self.F1 = FunctionSpace(N[1], 'F', dtype='d', domain=domain[1])
        self.D00 = FunctionSpace(N[0], family, bc=(0, 0), domain=domain[0])  # Streamwise velocity, not to be in tensorproductspace

        # Regular tensor product spaces
        self.TB = TensorProductSpace(comm, (self.B0, self.F1), modify_spaces_inplace=True) # Wall-normal velocity
        self.TD = TensorProductSpace(comm, (self.D0, self.F1), modify_spaces_inplace=True) # Streamwise velocity
        self.TC = TensorProductSpace(comm, (self.C0, self.F1), modify_spaces_inplace=True) # No bc
        self.TT = TensorProductSpace(comm, (self.T0, self.F1), modify_spaces_inplace=True) # Temperature
        self.BD = VectorSpace([self.TB, self.TD])  # Velocity vector space
        self.CD = VectorSpace([self.TD, self.TD])  # Convection vector space

        # Padded for dealiasing
        self.padding_factor = padding_factor
        self.TBp = self.TB.get_dealiased(padding_factor)
        self.TDp = self.TD.get_dealiased(padding_factor)
        self.BDp = self.BD.get_dealiased(padding_factor)

        self.u_ = Function(self.BD)      # Velocity solution
        self.uT_ = Function(self.BD)     # Velocity vector times T
        self.H_ = Function(self.CD)      # convection
        self.T_ = Function(self.TT)      # Temperature solution
        self.rhs_u = Function(self.CD)
        self.rhs_T = Function(self.CD)
        self.u00 = Function(self.D00)
        self.b0 = np.zeros((2,)+self.u00.shape)
        self.work = CachedArrayDict()

        self.file_u = ShenfunFile('_'.join((filename, 'U')), self.BD, backend='hdf5', mode='w', mesh='uniform')
        self.file_T = ShenfunFile('_'.join((filename, 'T')), self.TT, backend='hdf5', mode='w', mesh='uniform')

        self.mask = self.TB.get_mask_nyquist()
        self.K = self.TB.local_wavenumbers(scaled=True)
        self.X = self.TD.local_mesh(True)

        # Classes for fast projections
        self.dudx = Project(Dx(self.u_[0], 0, 1), self.TD)
        self.dudy = Project(Dx(self.u_[0], 1, 1), self.TB)
        self.dvdx = Project(Dx(self.u_[1], 0, 1), self.TC)
        self.dvdy = Project(Dx(self.u_[1], 1, 1), self.TD)
        self.curl = Project(Dx(self.u_[1], 0, 1) - Dx(self.u_[0], 1, 1), self.TC)
        self.divu = Project(div(self.u_), self.TD)

        # Create a checkpoint file used to restart simulations
        self.checkpoint = Checkpoint(filename,
                                     checkevery=checkpoint,
                                     data={'0': {'U': [self.u_],
                                                 'T': [self.T_]}})

    def initialize(self, rand=0.001, from_checkpoint=False):
        if from_checkpoint:
            self.checkpoint.read(self.u_, 'U', step=0)
            self.checkpoint.read(self.T_, 'T', step=0)
            self.checkpoint.open()
            tstep = self.checkpoint.f.attrs['tstep']
            t = self.checkpoint.f.attrs['t']
            self.checkpoint.close()
            self.update_bc(t)
            return t, tstep

        X = self.X
        funT = 1 if self.bcT[0] == 1 else 2
        fun = {1: 1,
               2: (0.9+0.1*np.sin(2*X[1]))}[funT]
        Tb = Array(self.TT)
        Tb[:] = 0.5*(1-X[0]+0.25*np.sin(np.pi*X[0]))*fun+rand*np.random.randn(*Tb.shape)*(1-X[0])*(1+X[0])
        self.T_ = Tb.forward(self.T_)
        self.T_.mask_nyquist(self.mask)
        return 0, 0

    def assemble(self):
        u = TrialFunction(self.TB)
        v = TestFunction(self.TB)
        p = TrialFunction(self.TT)
        q = TestFunction(self.TT)
        nu = self.nu
        dt = self.dt
        kappa = self.kappa

        # Chebyshev matrices are not sparse, so need a tailored solver. Legendre has simply 5 nonzero diagonals
        sol1 = chebyshev.la.Biharmonic if self.B0.family() == 'chebyshev' else la.SolverGeneric1ND
        sol2 = chebyshev.la.Helmholtz if self.B0.family() == 'chebyshev' else la.SolverGeneric1ND

        # u equation
        a, b = self.a, self.b
        self.solver = []
        self.Biharmonic_u = []
        for rk in range(3):
            mats = inner(v, div(grad(u)) - ((a[rk]+b[rk])*nu*dt/2.)*div(grad(div(grad(u)))))
            self.solver.append(sol1(mats))
            self.Biharmonic_u.append(Inner(v, div(grad(self.u_[0])) + ((a[rk]+b[rk])*nu*dt/2.)*div(grad(div(grad(self.u_[0]))))))
        self.convH = Inner(v, Dx(Dx(self.H_[1], 0, 1), 1, 1)-Dx(self.H_[0], 1, 2))
        self.d2Tdy2 = Inner(v, Dx(self.T_, 1, 2))

        # Note that linear forms use Inner, which creates a class that computes
        # the inner product with a fast matrix-vector product when called

        # T equation
        self.solverT = []
        self.rhsT = []
        for rk in range(3):
            matsT = inner(q, 2./(kappa*(a[rk]+b[rk])*dt)*p - div(grad(p)))
            self.solverT.append(sol2(matsT)) # boundary matrices are taken care of in solverT
            self.rhsT.append(Inner(q, 2./(self.kappa*(a[rk]+b[rk])*self.dt)*Expr(self.T_)+div(grad(self.T_))))
        self.div_uT = Inner(q, div(self.uT_))

        # v Eq for Fourier wavenumber 0
        u0 = TrialFunction(self.D00)
        v0 = TestFunction(self.D00)
        sol3 = chebyshev.la.Helmholtz if self.B0.family() == 'chebyshev' else la.Solver
        self.solver0 = []
        self.linear_rhs_u0 = []
        for rk in range(3):
            mats0 = inner(v0, 2./(nu*(a[rk]+b[rk])*dt)*u0 - div(grad(u0)))
            self.solver0.append(sol3(mats0))
            self.linear_rhs_u0.append(Inner(v0, 2./(nu*(a[rk]+b[rk])*dt)*Expr(self.u00)+div(grad(self.u00))))

    def update_bc(self, t):
        # Update time-dependent bcs.
        bc0 = self.T_.function_space().bases[0].bc
        bcp0 = self.T_.get_dealiased_space(self.padding_factor).bases[0].bc
        bc0.update_bcs_time(t)
        bcp0.update_bcs_time(t)

    def convection(self, u, H):
        up = u.backward(padding_factor=self.padding_factor)
        if self.conv == 0:
            dudxp = self.dudx().backward(padding_factor=self.padding_factor)
            dudyp = self.dudy().backward(padding_factor=self.padding_factor)
            dvdxp = self.dvdx().backward(padding_factor=self.padding_factor)
            dvdyp = self.dvdy().backward(padding_factor=self.padding_factor)
            H[0] = self.TDp.forward(up[0]*dudxp+up[1]*dudyp, H[0])
            H[1] = self.TDp.forward(up[0]*dvdxp+up[1]*dvdyp, H[1])
        elif self.conv == 1:
            curl = self.curl().backward(padding_factor=self.padding_factor)
            H[0] = self.TDp.forward(-curl*up[1])
            H[1] = self.TDp.forward(curl*up[0])
        H.mask_nyquist(self.mask)
        return H

    def compute_rhs_u(self, rhs, rk):
        a, b = self.a[rk], self.b[rk]
        self.H_ = self.convection(self.u_, self.H_)
        rhs[1] = 0
        rhs[1] += self.Biharmonic_u[rk]()
        w0 = self.convH()
        w1 = self.d2Tdy2()
        rhs[1] += a*self.dt*(w0+w1)
        rhs[1] += b*self.dt*rhs[0]
        rhs[0] = w0+w1
        rhs.mask_nyquist(self.mask)
        return rhs

    def compute_rhs_T(self, rhs, rk):
        a, b = self.a[rk], self.b[rk]
        rhs[1] = self.rhsT[rk]()
        up = self.u_.backward(padding_factor=self.padding_factor)
        T_p = self.T_.backward(padding_factor=self.padding_factor)
        self.uT_ = self.BDp.forward(up*T_p, self.uT_)
        w0 = self.div_uT()
        rhs[1] -= (2.*a/self.kappa/(a+b))*w0
        rhs[1] -= (2.*b/self.kappa/(a+b))*rhs[0]
        rhs[0] = w0
        rhs.mask_nyquist(self.mask)
        return rhs

    def compute_v(self, u, rk):
        if comm.Get_rank() == 0:
            self.u00[:] = u[1, :, 0].real

        # Find second velocity component
        with np.errstate(divide='ignore'):
            u[1] = 1j * self.dudx() / self.K[1]

        # Still have to compute for wavenumber = 0
        if comm.Get_rank() == 0:
            v0 = TestFunction(self.D00)
            w00 = self.work[(self.u00, 0, True)]
            a, b = self.a[rk], self.b[rk]
            self.b0[1] = self.linear_rhs_u0[rk]()
            w00 = inner(v0, self.H_[1, :, 0], output_array=w00)
            self.b0[1] -= (2./self.nu/(a+b))*(a*w00+b*self.b0[0])
            self.u00 = self.solver0[rk](self.b0[1], self.u00)
            u[1, :, 0] = self.u00
            self.b0[0] = w00
        return u

    def init_plots(self):
        self.ub = ub = self.u_.backward()
        self.T_b = T_b = self.T_.backward()
        if comm.Get_rank() == 0:
            plt.figure(1, figsize=(6, 3))
            self.im1 = plt.quiver(self.X[1][::4, ::4], self.X[0][::4, ::4], ub[1, ::4, ::4], ub[0, ::4, ::4], pivot='mid', scale=0.01)
            plt.draw()
            plt.figure(2, figsize=(6, 3))
            self.im2 = plt.contourf(self.X[1], self.X[0], T_b, 100)
            plt.draw()
            plt.pause(1e-6)

    def plot(self, t, tstep):
        if tstep % self.modplot == 0:
            ub = self.u_.backward(self.ub)
            e0 = dx(ub[0]*ub[0])
            e1 = dx(ub[1]*ub[1])
            T_b = self.T_.backward(self.T_b)
            e2 = inner(1, T_b*T_b)
            divu = self.divu().backward()
            e3 = dx(divu*divu)
            if comm.Get_rank() == 0:
                print("Time %2.5f Energy %2.6e %2.6e %2.6e div %2.6e" %(t, e0, e1, e2, e3))
                plt.figure(1)
                self.im1.set_UVC(ub[1, ::4, ::4], ub[0, ::4, ::4])
                self.im1.scale = np.linalg.norm(ub[1])
                plt.pause(1e-6)
                plt.figure(2)
                self.im2.axes.clear()
                self.im2.axes.contourf(self.X[1], self.X[0], T_b, 100)
                self.im2.autoscale()
                plt.pause(1e-6)

    def tofile(self, tstep):
        self.file_u.write(tstep, {'u': [self.u_.backward(mesh='uniform')]}, as_scalar=True)
        self.file_T.write(tstep, {'T': [self.T_.backward(mesh='uniform')],
                                  'curl': [self.curl().backward(mesh='uniform')]})

    def solve(self, t=0, tstep=0, end_time=1000):
        self.init_plots()
        while t < end_time-1e-8:
            for rk in range(3):
                rhs_u = self.compute_rhs_u(self.rhs_u, rk)   # rhs assembled for step k
                rhs_T = self.compute_rhs_T(self.rhs_T, rk)
                self.u_[0] = self.solver[rk](rhs_u[1], self.u_[0])
                if comm.Get_rank() == 0:
                    self.u_[0, :, 0] = 0
                u_ = self.compute_v(self.u_, rk)
                u_.mask_nyquist(self.mask)
                self.update_bc(t+self.dt*self.c[rk+1])       # Solving for step k+1
                T_ = self.solverT[rk](rhs_T[1], self.T_)     # This sets boundary dofs for T_
                T_.mask_nyquist(self.mask)
            t += self.dt
            self.end_of_tstep()
            tstep += 1
            self.plot(t, tstep)
            self.checkpoint.update(t, tstep)
            if tstep % self.modsave == 0:
                self.tofile(tstep)

if __name__ == '__main__':
    from time import time
    from mpi4py_fft import generate_xdmf
    t0 = time()
    d = {
        'N': (64, 128),
        'Ra': 100000.,
        'dt': 0.01,
        'filename': 'RB_64_128',
        'conv': 0,
        'modplot': 50,
        'modsave': 100,
        #'bcT': (0.9+0.1*sympy.sin(2*(y-tt)), 0),
        #'bcT': (0.9+0.1*sympy.sin(2*y), 0),
        'bcT': (1, 0),
        'family': 'C',
        'checkpoint': 100,
        #'padding_factor': 1
        }
    c = RayleighBenard(**d)
    t, tstep = c.initialize(rand=0.001, from_checkpoint=False)
    c.assemble()
    c.solve(t=t, tstep=tstep, end_time=0.1)
    print('Computing time %2.4f'%(time()-t0))
    if comm.Get_rank() == 0:
        generate_xdmf('_'.join((d['filename'], 'U'))+'.h5')
        generate_xdmf('_'.join((d['filename'], 'T'))+'.h5')
