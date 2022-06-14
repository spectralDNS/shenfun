from shenfun import *
np.warnings.filterwarnings('ignore')

class MicroPolar:
    """Micropolar channel flow solver

    """
    def __init__(self,
                 N=(32, 32, 32),
                 domain=((-1, 1), (0, 2*np.pi), (0, np.pi)),
                 Re=100,
                 J=1e-5,
                 m=0.1,
                 NP=8.3e4,
                 dt=0.001,
                 conv=0,
                 filename='MicroPolar',
                 family='C',
                 padding_factor=(1, 1.5, 1.5),
                 modplot=100,
                 modsave=1e8,
                 moderror=100,
                 checkpoint=1000,
                 sample_stats=1e8):
        self.Re = Re
        self.J = J
        self.m = m
        self.NP = NP
        self.dt = dt
        self.nu = 1/Re
        self.conv = conv
        self.modplot = modplot
        self.modsave = modsave
        self.moderror = moderror
        self.sample_stats = sample_stats
        self.filename = filename
        self.dpdy_source = -1
        self.im1 = None

        # Runge-Kutta 3 parameters
        self.a = (8./15., 5./12., 3./4.)
        self.b = (0.0, -17./60., -5./12.)
        self.c = (0.0, 8./15., 2./3., 1)

        # Regular spaces
        self.B0 = FunctionSpace(N[0], family, bc=(0, 0, 0, 0), domain=domain[0])
        self.D0 = FunctionSpace(N[0], family, bc=(0, 0), domain=domain[0])
        self.C0 = FunctionSpace(N[0], family, domain=domain[0])
        self.F1 = FunctionSpace(N[1], 'F', dtype='D', domain=domain[1])
        self.F2 = FunctionSpace(N[2], 'F', dtype='d', domain=domain[2])
        self.D00 = FunctionSpace(N[0], family, bc=(0, 0), domain=domain[0])  # Streamwise velocity, not to be in tensorproductspace
        self.C00 = self.D00.get_orthogonal()

        # Regular tensor product spaces
        # x, y, z is wall-normal, streamwise and spanwise, respectively
        self.TB = TensorProductSpace(comm, (self.B0, self.F1, self.F2), slab=True, modify_spaces_inplace=True) # Wall-normal velocity
        self.TD = TensorProductSpace(comm, (self.D0, self.F1, self.F2), slab=True, modify_spaces_inplace=True) # Streamwise velocity
        self.TC = TensorProductSpace(comm, (self.C0, self.F1, self.F2), slab=True, modify_spaces_inplace=True) # No bc
        self.BD = VectorSpace([self.TB, self.TD, self.TD])  # Velocity vector space
        self.CD = VectorSpace(self.TD)  # Convection vector space
        self.CC = VectorSpace([self.TD, self.TC, self.TC])  # Curl vector space
        self.WC = VectorSpace(self.TC)  # Curl curl vector space
        self.CR = CompositeSpace([self.TD, self.TD]) # For two steps in the RK3 integrator

        # Padded for dealiasing
        self.padding_factor = padding_factor
        self.TBp = self.TB.get_dealiased(padding_factor)
        self.TDp = self.TD.get_dealiased(padding_factor)
        self.BDp = self.BD.get_dealiased(padding_factor)

        self.u_ = Function(self.BD)      # Velocity solution
        self.w_ = Function(self.CD)      # Angular velocity solution
        self.H_ = Function(self.CD)      # convection
        self.HW_ = Function(self.CD)     # convection angular velocity
        self.curl = Function(self.CC)    # Velocity curl
        self.g_ = self.curl[0]           # g solution
        self.rhs_u = Function(self.CR)
        self.rhs_g = Function(self.CR)
        self.rhs_w0 = Function(self.CR)
        self.rhs_w1 = Function(self.CR)
        self.rhs_w2 = Function(self.CR)
        self.ub = Array(self.BD)
        self.wb = Array(self.CD)
        self.cb = Array(self.BD)

        self.v00 = Function(self.D00)   # For solving 1D problem for Fourier wavenumber 0, 0
        self.w00 = Function(self.D00)
        self.bv0 = np.zeros((2,)+self.v00.shape)
        self.bw0 = np.zeros((2,)+self.w00.shape)

        self.work = CachedArrayDict()
        self.mask = self.TB.get_mask_nyquist() # Used to set the Nyquist frequency to zero
        self.X = self.TD.local_mesh(bcast=True)
        self.K = self.TD.local_wavenumbers(scaled=True)

        # Classes for fast projections. All are not used except if self.conv=0
        self.dudx = Project(Dx(self.u_[0], 0, 1), self.TD)
        self.dudy = Project(Dx(self.u_[0], 1, 1), self.TB)
        self.dudz = Project(Dx(self.u_[0], 2, 1), self.TB)
        self.dvdx = Project(Dx(self.u_[1], 0, 1), self.TC)
        self.dvdy = Project(Dx(self.u_[1], 1, 1), self.TD)
        self.dvdz = Project(Dx(self.u_[1], 2, 1), self.TD)
        self.dwdx = Project(Dx(self.u_[2], 0, 1), self.TC)
        self.dwdy = Project(Dx(self.u_[2], 1, 1), self.TD)
        self.dwdz = Project(Dx(self.u_[2], 2, 1), self.TD)
        self.curly = Project(curl(self.u_)[1], self.TC, output_array=self.curl[1]) # curlx is already in g
        self.curlz = Project(curl(self.u_)[2], self.TC, output_array=self.curl[2])

        self.dw0dx = Project(Dx(self.w_[0], 0, 1), self.TC)
        self.dw0dy = Project(Dx(self.w_[0], 1, 1), self.TD)
        self.dw0dz = Project(Dx(self.w_[0], 2, 1), self.TD)
        self.dw1dx = Project(Dx(self.w_[1], 0, 1), self.TC)
        self.dw1dy = Project(Dx(self.w_[1], 1, 1), self.TD)
        self.dw1dz = Project(Dx(self.w_[1], 2, 1), self.TD)
        self.dw2dx = Project(Dx(self.w_[2], 0, 1), self.TC)
        self.dw2dy = Project(Dx(self.w_[2], 1, 1), self.TD)
        self.dw2dz = Project(Dx(self.w_[2], 2, 1), self.TD)
        self.curlwx = Project(curl(self.w_)[0], self.TD)
        self.curlcurlw = Project(curl(curl(self.w_))[0], self.TC)
        self.divu = Project(div(self.u_), self.TC)

        # File for storing the results
        self.file_u = ShenfunFile('_'.join((filename, 'U')), self.BD, backend='hdf5', mode='w', mesh='uniform')
        self.file_w = ShenfunFile('_'.join((filename, 'W')), self.CD, backend='hdf5', mode='w', mesh='uniform')

        # Create a checkpoint file used to restart simulations
        self.checkpoint = Checkpoint(filename,
                                     checkevery=checkpoint,
                                     data={'0': {'U': [self.u_], 'W': [self.w_]}})

    def init_from_checkpoint(self):
        self.checkpoint.read(self.u_, 'U', step=0)
        self.checkpoint.read(self.w_, 'W', step=0)
        self.g_[:] = 1j*self.K[1]*self.u_[2] - 1j*self.K[2]*self.u_[1]
        self.checkpoint.open()
        tstep = self.checkpoint.f.attrs['tstep']
        t = self.checkpoint.f.attrs['t']
        self.checkpoint.close()
        return t, tstep

    def assemble(self):
        u = TrialFunction(self.TB)
        v = TestFunction(self.TB)
        g = TrialFunction(self.TD)
        h = self.h = TestFunction(self.TD)

        nu = self.nu
        dt = self.dt

        # Chebyshev matrices are not sparse, so need a tailored solver. Legendre has simply 5 nonzero diagonals and can use generic solvers.
        sol1 = chebyshev.la.Biharmonic if self.B0.family() == 'chebyshev' else la.SolverGeneric1ND
        sol2 = chebyshev.la.Helmholtz if self.B0.family() == 'chebyshev' else la.SolverGeneric1ND

        # u equation
        a, b = self.a, self.b
        m = self.m
        self.solver = []
        self.linear_rhs_u = []
        for rk in range(3):
            mats = inner(v, div(grad(u)) - ((a[rk]+b[rk])*nu*dt/2.)*div(grad(div(grad(u)))))
            self.solver.append(sol1(mats))
            self.linear_rhs_u.append(Inner(v, div(grad(self.u_[0])) + ((a[rk]+b[rk])*nu*dt/2.)*div(grad(div(grad(self.u_[0]))))))
        cwx_ = self.curlwx.output_array
        self.nonlinear_rhs_u = Inner(v, Dx(Dx(self.H_[1], 0, 1), 1, 1)
                                       +Dx(Dx(self.H_[2], 0, 1), 2, 1)
                                       -Dx(self.H_[0], 1, 2)
                                       -Dx(self.H_[0], 2, 2))\
                             + Inner(v, m*nu*div(grad(cwx_)))

        # Note that linear forms use Inner, which creates a class that computes
        # the inner products with a fast matrix-vector product when called

        # g equation
        self.solverG = []
        self.linear_rhs_g = []
        for rk in range(3):
            matsG = inner(h, 2./(nu*(a[rk]+b[rk])*dt)*g - div(grad(g)))
            self.solverG.append(sol2(matsG))
            self.linear_rhs_g.append(Inner(h, 2./(nu*(a[rk]+b[rk])*dt)*Expr(self.g_)+div(grad(self.g_))))
        ccw_ = self.curlcurlw.output_array
        self.nonlinear_rhs_g = Inner(h, Dx(self.H_[1], 2, 1) - Dx(self.H_[2], 1, 1))\
                             + Inner(h, self.m*self.nu*Expr(ccw_))

        # v and w. Solve divergence constraint very fast for all wavenumbers except 0, 0
        K2 = self.K[1]*self.K[1]+self.K[2]*self.K[2]
        self.K_over_K2 = np.zeros((2,)+self.g_.shape)
        for i in range(2):
            self.K_over_K2[i] = self.K[i+1] / np.where(K2 == 0, 1, K2)

        if comm.Get_rank() == 0:
            # v and w. Momentum equation for Fourier wavenumber 0, 0
            u0 = TrialFunction(self.D00)
            v0 = TestFunction(self.D00)
            sol3 = chebyshev.la.Helmholtz if self.B0.family() == 'chebyshev' else la.Solver
            self.solver0 = []
            self.linear_rhs_v = []
            self.linear_rhs_w = []
            for rk in range(3):
                mats0 = inner(v0, 2./(nu*(a[rk]+b[rk])*dt)*u0 - div(grad(u0)))
                self.solver0.append(sol3(mats0))
                self.linear_rhs_v.append(Inner(v0, 2./(nu*(a[rk]+b[rk])*dt)*Expr(self.v00)+div(grad(self.v00))))
                self.linear_rhs_w.append(Inner(v0, 2./(nu*(a[rk]+b[rk])*dt)*Expr(self.w00)+div(grad(self.w00))))
            self.h1 = Function(self.D00)  # Need to copy H_[1, :, 0, 0] into this Function before calling nonlinear_rhs_v
            self.wz = Function(self.D00)  # Need copy of self.w_[2, :, 0, 0]
            self.nonlinear_rhs_v = Inner(v0, -Expr(self.h1)) \
                                  +Inner(v0, -m*nu*Dx(self.wz, 0, 1))
            self.h2 = Function(self.D00)  # Copy of H_[2, :, 0, 0]
            self.wy = Function(self.D00)  # Copy self.w_[1, :, 0, 0]
            self.nonlinear_rhs_w = Inner(v0, -Expr(self.h2)) \
                                  +Inner(v0, m*nu*Dx(self.wy, 0, 1))
            source = Array(self.D00)
            source[:] = -self.dpdy_source # dpdy_source set by subclass
            self.dpdy = inner(v0, source)

        # Angular momentum equations
        self.solverW = []
        self.linear_rhs_w0 = []
        self.linear_rhs_w1 = []
        self.linear_rhs_w2 = []
        self.kappa = kappa = self.m/self.J/self.NP/self.Re
        NP = self.NP
        for rk in range(3):
            matsG = inner(h, (2./(kappa*(a[rk]+b[rk])*dt)+2*NP)*g - div(grad(g)))
            self.solverW.append(sol2(matsG)) # boundary matrices are taken care of in solverT
            self.linear_rhs_w0.append(Inner(h, (2./(kappa*(a[rk]+b[rk])*dt)-2*NP)*Expr(self.w_[0]) + div(grad(self.w_[0]))))
            self.linear_rhs_w1.append(Inner(h, (2./(kappa*(a[rk]+b[rk])*dt)-2*NP)*Expr(self.w_[1]) + div(grad(self.w_[1]))))
            self.linear_rhs_w2.append(Inner(h, (2./(kappa*(a[rk]+b[rk])*dt)-2*NP)*Expr(self.w_[2]) + div(grad(self.w_[2]))))

    def convection(self, u, w, H, HW):
        up = u.backward(padding_factor=self.padding_factor)

        if self.conv == 0:
            # Standard convection
            dudxp = self.dudx().backward(padding_factor=self.padding_factor)
            dudyp = self.dudy().backward(padding_factor=self.padding_factor)
            dudzp = self.dudz().backward(padding_factor=self.padding_factor)
            dvdxp = self.dvdx().backward(padding_factor=self.padding_factor)
            dvdyp = self.dvdy().backward(padding_factor=self.padding_factor)
            dvdzp = self.dvdz().backward(padding_factor=self.padding_factor)
            dwdxp = self.dwdx().backward(padding_factor=self.padding_factor)
            dwdyp = self.dwdy().backward(padding_factor=self.padding_factor)
            dwdzp = self.dwdz().backward(padding_factor=self.padding_factor)
            H[0] = self.TDp.forward(up[0]*dudxp+up[1]*dudyp+up[2]*dudzp, H[0])
            H[1] = self.TDp.forward(up[0]*dvdxp+up[1]*dvdyp+up[2]*dvdzp, H[1])
            H[2] = self.TDp.forward(up[0]*dwdxp+up[1]*dwdyp+up[2]*dwdzp, H[2])
        elif self.conv == 1:
            # Vortex form
            self.curly() # Compute y-component of curl. Stored in self.curl[1]
            self.curlz() # Compute y-component of curl. Stored in self.curl[2]
            curl = self.curl.backward(padding_factor=self.padding_factor)
            cb = self.work[(up, 1, True)]
            cb = cross(cb, curl, up)
            H[0] = self.TDp.forward(cb[0], H[0])
            H[1] = self.TDp.forward(cb[1], H[1])
            H[2] = self.TDp.forward(cb[2], H[2])

        dw0dxp = self.dw0dx().backward(padding_factor=self.padding_factor)
        dw0dyp = self.dw0dy().backward(padding_factor=self.padding_factor)
        dw0dzp = self.dw0dz().backward(padding_factor=self.padding_factor)
        dw1dxp = self.dw1dx().backward(padding_factor=self.padding_factor)
        dw1dyp = self.dw1dy().backward(padding_factor=self.padding_factor)
        dw1dzp = self.dw1dz().backward(padding_factor=self.padding_factor)
        dw2dxp = self.dw2dx().backward(padding_factor=self.padding_factor)
        dw2dyp = self.dw2dy().backward(padding_factor=self.padding_factor)
        dw2dzp = self.dw2dz().backward(padding_factor=self.padding_factor)
        HW[0] = self.TDp.forward(up[0]*dw0dxp+up[1]*dw0dyp+up[2]*dw0dzp, HW[0])
        HW[1] = self.TDp.forward(up[0]*dw1dxp+up[1]*dw1dyp+up[2]*dw1dzp, HW[1])
        HW[2] = self.TDp.forward(up[0]*dw2dxp+up[1]*dw2dyp+up[2]*dw2dzp, HW[2])
        H.mask_nyquist(self.mask)
        HW.mask_nyquist(self.mask)

    def compute_rhs_u(self, rhs, rk):
        a, b = self.a[rk], self.b[rk]
        self.convection(self.u_, self.w_, self.H_, self.HW_)
        rhs[1] = 0
        rhs[1] += self.linear_rhs_u[rk]()
        self.curlwx()
        w0 = self.nonlinear_rhs_u()
        rhs[1] += self.dt*(a*w0+b*rhs[0])
        rhs[0] = w0
        rhs.mask_nyquist(self.mask)
        return rhs

    def compute_rhs_g(self, rhs, rk):
        a, b = self.a[rk], self.b[rk]
        rhs[1] = self.linear_rhs_g[rk]()
        self.curlcurlw()
        hg = self.nonlinear_rhs_g()
        rhs[1] += (2./self.nu/(a+b))*(a*hg+b*rhs[0])
        rhs[0] = hg
        rhs.mask_nyquist(self.mask)
        return rhs

    def compute_rhs_w0(self, rhs, rk):
        a, b = self.a[rk], self.b[rk]
        rhs[1] = self.linear_rhs_w0[rk]()
        hg = inner(self.h, -self.HW_[0].backward()+self.kappa*self.NP*self.curl[0].backward())
        rhs[1] += (2./self.kappa/(a+b))*(a*hg+b*rhs[0])
        rhs[0] = hg
        rhs.mask_nyquist(self.mask)
        return rhs

    def compute_rhs_w1(self, rhs, rk):
        a, b = self.a[rk], self.b[rk]
        rhs[1] = self.linear_rhs_w1[rk]()
        hg = inner(self.h, -self.HW_[1].backward()+self.kappa*self.NP*self.curl[1].backward())
        rhs[1] += (2./self.kappa/(a+b))*(a*hg+b*rhs[0])
        rhs[0] = hg
        rhs.mask_nyquist(self.mask)
        return rhs

    def compute_rhs_w2(self, rhs, rk):
        a, b = self.a[rk], self.b[rk]
        rhs[1] = self.linear_rhs_w2[rk]()
        hg = inner(self.h, -self.HW_[2].backward()+self.kappa*self.NP*self.curl[2].backward())
        rhs[1] += (2./self.kappa/(a+b))*(a*hg+b*rhs[0])
        rhs[0] = hg
        rhs.mask_nyquist(self.mask)
        return rhs

    def compute_vw(self, u, rk):
        if comm.Get_rank() == 0:
            self.v00[:] = u[1, :, 0, 0].real
            self.w00[:] = u[2, :, 0, 0].real

        # Find velocity components v and w
        f = self.dudx() # Note paper uses f=-dudx
        g = self.g_
        u[1] = 1j*(self.K_over_K2[0]*f + self.K_over_K2[1]*g)
        u[2] = 1j*(self.K_over_K2[1]*f - self.K_over_K2[0]*g)

        # Still have to compute for wavenumber = 0, 0
        if comm.Get_rank() == 0:
            w_ = self.work[(self.v00, 0, True)]
            a, b = self.a[rk], self.b[rk]

            # v component
            w_[:] = 0
            self.h1[:] = self.H_[1, :, 0, 0].real
            self.wz[:] = self.w_[2, :, 0, 0].real
            w_ = self.nonlinear_rhs_v()
            w_ -= self.dpdy
            self.bv0[1] = self.linear_rhs_v[rk]()
            self.bv0[1] += (2./self.nu/(a+b))*(a*w_+b*self.bv0[0])
            self.v00 = self.solver0[rk](self.bv0[1], self.v00)
            u[1, :, 0, 0] = self.v00
            self.bv0[0] = w_

            # w component
            w_[:] = 0
            self.h2[:] = self.H_[2, :, 0, 0].real
            self.wy[:] = self.w_[1, :, 0, 0].real
            w_ = self.nonlinear_rhs_w()
            self.bw0[1] = self.linear_rhs_w[rk]()
            self.bw0[1] += (2./self.nu/(a+b))*(a*w_+b*self.bw0[0])
            self.w00 = self.solver0[rk](self.bw0[1], self.w00)
            u[2, :, 0, 0] = self.w00
            self.bw0[0] = w_

        return u

    def initialize(self, from_checkpoint=False):
        pass

    def plot(self, t, tstep):
        pass

    def update(self, t, tstep):
        self.plot(t, tstep)
        self.print_energy_and_divergence(t, tstep)

    def tofile(self, tstep):
        self.file_u.write(tstep, {'u': [self.u_.backward(mesh='uniform')]}, as_scalar=True)
        self.file_w.write(tstep, {'w': [self.w_.backward(mesh='uniform')]}, as_scalar=True)

    def solve(self, t=0, tstep=0, end_time=1000):
        self.update(t, tstep)
        while t < end_time-1e-8:
            for rk in range(3):
                rhs_u = self.compute_rhs_u(self.rhs_u, rk)   # rhs assembled for step k
                rhs_g = self.compute_rhs_g(self.rhs_g, rk)
                rhs_w0 = self.compute_rhs_w0(self.rhs_w0, rk)
                rhs_w1 = self.compute_rhs_w1(self.rhs_w1, rk)
                rhs_w2 = self.compute_rhs_w2(self.rhs_w2, rk)
                self.u_[0] = self.solver[rk](rhs_u[1], self.u_[0])
                self.g_ = self.solverG[rk](rhs_g[1], self.g_)
                self.u_ = self.compute_vw(self.u_, rk)
                self.w_[0] = self.solverW[rk](rhs_w0[1], self.w_[0])
                self.w_[1] = self.solverW[rk](rhs_w1[1], self.w_[1])
                self.w_[2] = self.solverW[rk](rhs_w2[1], self.w_[2])

            t += self.dt
            tstep += 1
            self.update(t, tstep)
            self.checkpoint.update(t, tstep)
            if tstep % self.modsave == 0:
                self.tofile(tstep)
