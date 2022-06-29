from warnings import WarningMessage
from shenfun import *
np.warnings.filterwarnings('ignore')

class KMM:
    """Navier Stokes channel flow solver

    The wall normal direction is along the x-axis, the streamwise along the y-axis
    and the spanwise direction is along the z-axis.

    The solver is fully spectral with Chebyshev (or Legendre) in the wall-normal
    direction and Fourier in the remaining 2.

    Using the equations described by Kim, Moser, Moin (https://doi.org/10.1017/S0022112087000892)
    but with the spectral Galerkin method in space and a chosen time stepper.

    Parameters
    ----------
    N : 3-tuple of ints
        The global shape in physical space (quadrature points)
    domain : 3-tuple of 2-tuples
        The size of the three domains
    nu : Viscosity coefficient
    dt : Timestep
    conv : Choose convection method
        - 0 - Standard convection
        - 1 - Vortex type
    filename : str, optional
        Filenames are started with this name
    family : str, optional
        Chebyshev is normal, but Legendre works as well
    padding_factor : 3-tuple of numbers, optional
        For dealiasing, backward transforms to real space are
        padded with zeros in spectral space using these many points
    modplot : int, optional
        Plot some results every modplot timestep. If negative, no plotting
    modsave : int, optional
        Save results to hdf5 every modsave timestep.
    moderror : int, optional
        Print diagnostics every moderror timestep
    checkpoint : int, optional
        Save required data for restart to hdf5 every checkpoint timestep.
    timestepper : str, optional
        Choose timestepper
    Note
    ----
    Simulations may be killed gracefully by placing a file named 'killshenfun'
    in the folder running the solver from. The solver will then first store
    the results by checkpointing, before exiting.

    """
    def __init__(self,
                 N=(32, 32, 32),
                 domain=((-1, 1), (0, 2*np.pi), (0, np.pi)),
                 nu=0.01,
                 dt=0.1,
                 conv=0,
                 dpdy=1,
                 filename='KMM',
                 family='C',
                 padding_factor=(1, 1.5, 1.5),
                 modplot=100,
                 modsave=1e8,
                 moderror=100,
                 checkpoint=1000,
                 timestepper='IMEXRK3'):
        self.N = N
        self.nu = nu
        self.dt = dt
        self.conv = conv
        self.modplot = modplot
        self.modsave = modsave
        self.moderror = moderror
        self.filename = filename
        self.padding_factor = padding_factor
        self.dpdy = dpdy
        self.PDE = PDE = globals().get(timestepper)
        self.im1 = None

        # Regular spaces
        self.B0 = FunctionSpace(N[0], family, bc=(0, 0, 0, 0), domain=domain[0])
        self.D0 = FunctionSpace(N[0], family, bc=(0, 0), domain=domain[0])
        self.C0 = FunctionSpace(N[0], family, domain=domain[0])
        self.F1 = FunctionSpace(N[1], 'F', dtype='D', domain=domain[1])
        self.F2 = FunctionSpace(N[2], 'F', dtype='d', domain=domain[2])
        self.D00 = FunctionSpace(N[0], family, bc=(0, 0), domain=domain[0])  # Streamwise velocity, not to be in tensorproductspace
        self.C00 = self.D00.get_orthogonal()

        # Regular tensor product spaces
        self.TB = TensorProductSpace(comm, (self.B0, self.F1, self.F2), collapse_fourier=False, slab=True, modify_spaces_inplace=True) # Wall-normal velocity
        self.TD = TensorProductSpace(comm, (self.D0, self.F1, self.F2), collapse_fourier=False, slab=True, modify_spaces_inplace=True) # Streamwise velocity
        self.TC = TensorProductSpace(comm, (self.C0, self.F1, self.F2), collapse_fourier=False, slab=True, modify_spaces_inplace=True) # No bc
        self.BD = VectorSpace([self.TB, self.TD, self.TD])  # Velocity vector space
        self.CD = VectorSpace(self.TD)                      # Convection vector space
        self.CC = VectorSpace([self.TD, self.TC, self.TC])  # Curl vector space

        # Padded space for dealiasing
        self.TDp = self.TD.get_dealiased(padding_factor)

        self.u_ = Function(self.BD)      # Velocity vector solution
        self.H_ = Function(self.CD)      # convection
        self.curl = Function(self.CC)    # Velocity curl
        self.g_ = self.curl[0]           # g solution is first component of curl
        self.ub = Array(self.BD)

        self.v00 = Function(self.D00)   # For solving 1D problem for Fourier wavenumber 0, 0
        self.w00 = Function(self.D00)

        self.work = CachedArrayDict()
        self.mask = self.TB.get_mask_nyquist() # Used to set the Nyquist frequency to zero
        self.X = self.TD.local_mesh(bcast=True)
        self.K = self.TD.local_wavenumbers(scaled=True)
        self.solP = None

        # Classes for fast projections. All are not used except if self.conv=0
        self.dudx = Project(Dx(self.u_[0], 0, 1), self.TD)
        if self.conv == 0:
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
        self.divu = Project(div(self.u_), self.TC)

        # File for storing the results
        self.file_u = ShenfunFile('_'.join((filename, 'U')), self.BD, backend='hdf5', mode='w', mesh='uniform')

        # Create a checkpoint file used to restart simulations
        self.checkpoint = Checkpoint(filename,
                                     checkevery=checkpoint,
                                     data={'0': {'U': [self.u_]}})

        # set up equations
        v = TestFunction(self.TB)
        h = TestFunction(self.TD)

        # Chebyshev matrices are not sparse, so need a tailored solver. Legendre has simply 5 nonzero diagonals and can use generic solvers.
        sol1 = chebyshev.la.Biharmonic if self.B0.family() == 'chebyshev' else la.SolverGeneric1ND
        sol2 = chebyshev.la.Helmholtz if self.B0.family() == 'chebyshev' else la.SolverGeneric1ND

        self.pdes = {

            'u': PDE(v,                                   # test function
                     div(grad(self.u_[0])),               # u
                     lambda f: self.nu*div(grad(f)),      # linear operator on u
                     Dx(Dx(self.H_[1], 0, 1), 1, 1)+Dx(Dx(self.H_[2], 0, 1), 2, 1)-Dx(self.H_[0], 1, 2)-Dx(self.H_[0], 2, 2),
                     dt=self.dt,
                     solver=sol1,
                     latex=r"\frac{\partial \nabla^2 u}{\partial t} = \nu \nabla^4 u + \frac{\partial^2 N_y}{\partial x \partial y} + \frac{\partial^2 N_z}{\partial x \partial z} - \frac{\partial^2 N_x}{\partial y^2} - \frac{\partial^2 N_x}{\partial z^2}"),

            'g': PDE(h,
                     self.g_,
                     lambda f: self.nu*div(grad(f)),
                     Dx(self.H_[1], 2, 1) - Dx(self.H_[2], 1, 1),
                     dt=self.dt,
                     solver=sol2,
                     latex=r"\frac{\partial g}{\partial t} = \nu \nabla^2 g + \frac{\partial N_y}{\partial z} - \frac{\partial N_z}{\partial y}")
        }

        # v and w. Solve divergence constraint for all wavenumbers except 0, 0
        r""":math:`\nabla \cdot \vec{u} = 0`"""
        K2 = self.K[1]*self.K[1]+self.K[2]*self.K[2]
        self.K_over_K2 = np.zeros((2,)+self.g_.shape)
        for i in range(2):
            self.K_over_K2[i] = self.K[i+1] / np.where(K2 == 0, 1, K2)

        # v and w. Momentum equation for Fourier wavenumber 0, 0
        if comm.Get_rank() == 0:
            v0 = TestFunction(self.D00)
            self.h1 = Function(self.D00)  # Copy from H_[1, :, 0, 0] (cannot use view since not contiguous)
            self.h2 = Function(self.D00)  # Copy from H_[2, :, 0, 0]
            source = Array(self.C00)
            source[:] = -self.dpdy        # dpdy set by subclass
            sol = chebyshev.la.Helmholtz if self.B0.family() == 'chebyshev' else la.Solver
            self.pdes1d = {
                'v0': PDE(v0,
                          self.v00,
                          lambda f: self.nu*div(grad(f)),
                          [-Expr(self.h1), source],
                          dt=self.dt,
                          solver=sol,
                          latex=r"\frac{\partial v}{\partial t} = \nu \frac{\partial^2 v}{\partial x^2} - N_y - \frac{\partial p}{\partial y}"),

                'w0': PDE(v0,
                          self.w00,
                          lambda f: self.nu*div(grad(f)),
                          -Expr(self.h2),
                          dt=self.dt,
                          solver=sol,
                          latex=r"\frac{\partial w}{\partial t} = \nu \frac{\partial^2 w}{\partial x^2} - N_z")
            }

    def convection(self):
        H = self.H_.v # .v to access numpy array directly for faster lookup
        self.up = self.u_.backward(padding_factor=self.padding_factor)
        up = self.up.v
        if self.conv == 0:
            dudxp = self.dudx().backward(padding_factor=self.padding_factor).v
            dudyp = self.dudy().backward(padding_factor=self.padding_factor).v
            dudzp = self.dudz().backward(padding_factor=self.padding_factor).v
            dvdxp = self.dvdx().backward(padding_factor=self.padding_factor).v
            dvdyp = self.dvdy().backward(padding_factor=self.padding_factor).v
            dvdzp = self.dvdz().backward(padding_factor=self.padding_factor).v
            dwdxp = self.dwdx().backward(padding_factor=self.padding_factor).v
            dwdyp = self.dwdy().backward(padding_factor=self.padding_factor).v
            dwdzp = self.dwdz().backward(padding_factor=self.padding_factor).v
            H[0] = self.TDp.forward(up[0]*dudxp+up[1]*dudyp+up[2]*dudzp, H[0])
            H[1] = self.TDp.forward(up[0]*dvdxp+up[1]*dvdyp+up[2]*dvdzp, H[1])
            H[2] = self.TDp.forward(up[0]*dwdxp+up[1]*dwdyp+up[2]*dwdzp, H[2])
        elif self.conv == 1:
            self.curly() # Compute y-component of curl. Stored in self.curl[1]
            self.curlz() # Compute y-component of curl. Stored in self.curl[2]
            curl = self.curl.backward(padding_factor=self.padding_factor).v
            cb = self.work[(up, 1, True)]
            cb = cross(cb, curl, up)
            H[0] = self.TDp.forward(cb[0], H[0])
            H[1] = self.TDp.forward(cb[1], H[1])
            H[2] = self.TDp.forward(cb[2], H[2])
        self.H_.mask_nyquist(self.mask)

    def compute_vw(self, rk):
        u = self.u_.v
        if comm.Get_rank() == 0:
            self.v00[:] = u[1, :, 0, 0].real
            self.w00[:] = u[2, :, 0, 0].real
            self.h1[:] = self.H_[1, :, 0, 0].real
            self.h2[:] = self.H_[2, :, 0, 0].real

        # Find velocity components v and w from f, g and div. constraint
        f = self.dudx() # Note paper uses f=-dudx
        g = self.g_
        u[1] = 1j*(self.K_over_K2[0]*f + self.K_over_K2[1]*g)
        u[2] = 1j*(self.K_over_K2[1]*f - self.K_over_K2[0]*g)

        # Still have to compute for wavenumber = 0, 0
        if comm.Get_rank() == 0:
            # v component
            self.pdes1d['v0'].compute_rhs(rk)
            u[1, :, 0, 0] = self.pdes1d['v0'].solve_step(rk)

            # w component
            self.pdes1d['w0'].compute_rhs(rk)
            u[2, :, 0, 0] = self.pdes1d['w0'].solve_step(rk)

        return u

    def compute_pressure(self):
        if self.solP is None:
            # The boundary condition of p is dp/dn=nu*d^2u/dn^2
            self.d2udx2 = Project(self.nu*Dx(self.u_[0], 0, 2), self.TC)
            N0 = self.N0 = FunctionSpace(self.N[0], self.B0.family(), bc={'left': {'N': self.d2udx2()}, 'right': {'N': self.d2udx2()}})
            TN = self.TN = TensorProductSpace(comm, (N0, self.F1, self.F2), collapse_fourier=False, slab=True, modify_spaces_inplace=True)
            sol = chebyshev.la.Helmholtz if self.B0.family() == 'chebyshev' else la.SolverGeneric1ND
            self.divH = Inner(TestFunction(TN), -div(self.H_))
            self.solP = sol(inner(TestFunction(TN), div(grad(TrialFunction(TN)))))
            self.p_ = Function(TN)

        self.d2udx2()
        self.N0.bc.set_tensor_bcs(self.N0, self.TN)
        p_ = self.solP(self.divH(), self.p_, constraints=((0, 0, 0),))
        return p_

    def print_energy_and_divergence(self, t, tstep):
        if tstep % self.moderror == 0 and self.moderror > 0:
            ub = self.u_.backward(self.ub)
            e0 = inner(1, ub[0]*ub[0])
            e1 = inner(1, ub[1]*ub[1])
            e2 = inner(1, ub[2]*ub[2])
            divu = self.divu().backward()
            e3 = np.sqrt(inner(1, divu*divu))
            if comm.Get_rank() == 0:
                print("Time %2.5f Energy %2.6e %2.6e %2.6e div %2.6e" %(t, e0, e1, e2, e3))

    def init_from_checkpoint(self):
        self.checkpoint.read(self.u_, 'U', step=0)
        self.g_[:] = 1j*self.K[1]*self.u_[2] - 1j*self.K[2]*self.u_[1]
        self.checkpoint.open()
        tstep = self.checkpoint.f.attrs['tstep']
        t = self.checkpoint.f.attrs['t']
        self.checkpoint.close()
        self.convection()
        for pde in self.pdes.values():
            pde.initialize()
        for pde in self.pdes1d.values():
            pde.initialize()
        return t, tstep

    def initialize(self, from_checkpoint=False):
        if from_checkpoint:
            return self.init_from_checkpoint()
        raise RuntimeError('Initialize solver in subclass')

    def plot(self, t, tstep):
        pass

    def update(self, t, tstep):
        self.plot(t, tstep)
        self.print_energy_and_divergence(t, tstep)

    def tofile(self, tstep):
        self.file_u.write(tstep, {'u': [self.u_.backward(mesh='uniform')]}, as_scalar=True)

    def prepare_step(self, rk):
        self.convection()

    def assemble(self):
        for pde in self.pdes.values():
            pde.assemble()
        if comm.Get_rank() == 0:
            for pde in self.pdes1d.values():
                pde.assemble()

    def solve(self, t=0, tstep=0, end_time=1000):
        self.assemble()
        while t < end_time-1e-8:
            for rk in range(self.PDE.steps()):
                self.prepare_step(rk)
                for eq in self.pdes.values():
                    eq.compute_rhs(rk)
                for eq in self.pdes.values():
                    eq.solve_step(rk)
                self.compute_vw(rk)
            t += self.dt
            tstep += 1
            self.update(t, tstep)
            self.checkpoint.update(t, tstep)
            if tstep % self.modsave == 0:
                self.tofile(tstep)
