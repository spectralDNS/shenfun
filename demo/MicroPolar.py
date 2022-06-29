from random import sample
from shenfun import *
from ChannelFlow import KMM
np.warnings.filterwarnings('ignore')

class MicroPolar(KMM):
    """Micropolar channel flow solver

    Parameters
    ----------
    N : 3-tuple of ints
        The global shape in physical space (quadrature points)
    domain : 3-tuple of 2-tuples
        The size of the three domains
    Re : Reynolds number
    J : number
        model parameter
    m : number
        model parameter
    NP : number
        model parameter
    dt : Timestep
    conv : Choose velocity convection method
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
        Save results to hdf5 every modsave timestep
    moderror : int, optional
        Print diagnostics every moderror timestep
    checkpoint : int, optional
        Save required data for restart to hdf5 every checkpoint timestep
    sample_stats : int, optional
        Sample statistics every sample_stats timestep
    timestepper : str, optional
        Choose timestepper
        - 'IMEXRK222'
        - 'IMEXRK3'
        - 'IMEXRK443'

    Note
    ----
    Simulations may be killed gracefully by placing a file named 'killshenfun'
    in the folder running the solver from. The solver will then first store
    the results by checkpointing, before exiting.

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
                 timestepper='IMEXRK3'):
        utau = self.utau = 1
        KMM.__init__(self, N=N, domain=domain, nu=1/Re, dt=dt, conv=conv,
                     filename=filename, family=family, padding_factor=padding_factor,
                     modplot=modplot, modsave=modsave, moderror=moderror, dpdy=-utau**2,
                     checkpoint=checkpoint, timestepper=timestepper)
        self.Re = Re
        self.J = J
        self.m = m
        self.NP = NP

        # New spaces and Functions used by micropolar model
        self.WC = VectorSpace(self.TC)   # Curl curl vector space
        self.w_ = Function(self.CD)      # Angular velocity solution
        self.HW_ = Function(self.CD)     # convection angular velocity
        self.wz = Function(self.D00)
        self.wy = Function(self.D00)
        self.ub = Array(self.BD)
        self.wb = Array(self.CD)
        self.cb = Array(self.BD)

        # Classes for fast projections used by convection
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
        self.curlcurlwx = Project(curl(curl(self.w_))[0], self.TC)

        # File for storing the results
        self.file_w = ShenfunFile('_'.join((filename, 'W')), self.CD, backend='hdf5', mode='w', mesh='uniform')

        # Create a checkpoint file used to restart simulations
        self.checkpoint.data['0']['W'] = [self.w_]

        h = TestFunction(self.TD)

        # Chebyshev matrices are not sparse, so need a tailored solver. Legendre has simply 5 nonzero diagonals and can use generic solvers.
        sol2 = chebyshev.la.Helmholtz if self.B0.family() == 'chebyshev' else la.SolverGeneric1ND

        # Modify u equation
        nu = self.nu
        cwx_ = self.curlwx.output_array
        self.pdes['u'].N = [self.pdes['u'].N, m*nu*div(grad(cwx_))]
        self.pdes['u'].latex += r'+m \nu \nabla^2 (\nabla \times \vec{w})_x'

        # Modify g equation
        ccw_ = self.curlcurlwx.output_array
        self.pdes['g'].N = [self.pdes['g'].N, m*nu*Expr(ccw_)]
        self.pdes['g'].latex += r'+m \nu (\nabla \times \nabla \times \vec{w})_x'

        if comm.Get_rank() == 0:
            # Modify v0 and w0 equations
            self.pdes1d['v0'].N.append(-m*nu*Dx(self.wz, 0, 1))
            self.pdes1d['w0'].N = [self.pdes1d['w0'].N, m*nu*Dx(self.wy, 0, 1)]
            self.pdes1d['v0'].latex += r'-m \nu \frac{\partial w_z}{\partial x}'
            self.pdes1d['w0'].latex += r'+m \nu \frac{\partial w_y}{\partial x}'

        # Angular momentum equations
        self.kappa = kappa = m/J/NP/Re
        self.pdes['w0'] = self.PDE(h,
                                   self.w_[0],
                                   lambda f: kappa*div(grad(f))-2*NP*kappa*f,
                                   [-Expr(self.HW_[0]), kappa*NP*Expr(self.curl[0])],
                                   dt=self.dt,
                                   solver=sol2,
                                   latex=r"\frac{\partial w_x}{\partial t} +\vec{u} \cdot \nabla w_x = \kappa \nabla^2 w_x + \kappa N (\nabla \times \vec{u})_x")

        self.pdes['w1'] = self.PDE(h,
                                   self.w_[1],
                                   lambda f: kappa*div(grad(f))-2*NP*kappa*f,
                                   [-Expr(self.HW_[1]), kappa*NP*Expr(self.curl[1])],
                                   dt=self.dt,
                                   solver=sol2,
                                   latex=r"\frac{\partial w_y}{\partial t} +\vec{u} \cdot \nabla w_y = \kappa \nabla^2 w_y + \kappa N (\nabla \times \vec{u})_y")

        self.pdes['w2'] = self.PDE(h,
                                   self.w_[2],
                                   lambda f: kappa*div(grad(f))-2*NP*kappa*f,
                                   [-Expr(self.HW_[2]), kappa*NP*Expr(self.curl[2])],
                                   dt=self.dt,
                                   solver=sol2,
                                   latex=r"\frac{\partial w_z}{\partial t} +\vec{u} \cdot \nabla w_z = \kappa \nabla^2 w_z + \kappa N (\nabla \times \vec{u})_z")

    def init_from_checkpoint(self):
        self.checkpoint.read(self.u_, 'U', step=0)
        self.checkpoint.read(self.w_, 'W', step=0)
        self.g_[:] = 1j*self.K[1]*self.u_[2] - 1j*self.K[2]*self.u_[1]
        self.checkpoint.open()
        tstep = self.checkpoint.f.attrs['tstep']
        t = self.checkpoint.f.attrs['t']
        self.checkpoint.close()
        return t, tstep

    def convection(self):
        KMM.convection(self)
        HW = self.HW_
        up = self.up
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
        HW.mask_nyquist(self.mask)

    def tofile(self, tstep):
        self.file_u.write(tstep, {'u': [self.u_.backward(mesh='uniform')]}, as_scalar=True)
        self.file_w.write(tstep, {'w': [self.w_.backward(mesh='uniform')]}, as_scalar=True)

    def compute_vw(self, rk):
        if comm.Get_rank() == 0:
            self.wy[:] = self.w_[1, :, 0, 0].real
            self.wz[:] = self.w_[2, :, 0, 0].real
        KMM.compute_vw(self, rk)
