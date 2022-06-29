from shenfun import *
from ChannelFlow import KMM
import matplotlib.pyplot as plt
import sympy
np.warnings.filterwarnings('ignore')

# pylint: disable=attribute-defined-outside-init

x, y, tt = sympy.symbols('x,y,t', real=True)

comm = MPI.COMM_WORLD

class RayleighBenard(KMM):

    def __init__(self,
                 N=(32, 32, 32),
                 domain=((-1, 1), (0, 2*np.pi), (0, 2*np.pi)),
                 Ra=10000.,
                 Pr=0.7,
                 dt=0.1,
                 bcT=(0, 1),
                 conv=0,
                 filename='RB',
                 family='C',
                 padding_factor=(1, 1.5, 1.5),
                 modplot=100,
                 modsave=1e8,
                 moderror=100,
                 checkpoint=1000,
                 timestepper='IMEXRK3'):
        KMM.__init__(self, N=N, domain=domain, nu=np.sqrt(Pr/Ra), dt=dt, conv=conv,
                     filename=filename, family=family, padding_factor=padding_factor,
                     modplot=modplot, modsave=modsave, moderror=moderror,
                     checkpoint=checkpoint, timestepper=timestepper, dpdy=0)
        self.kappa = 1./np.sqrt(Pr*Ra)
        self.bcT = bcT

        # Additional spaces and functions for Temperature equation
        self.T0 = FunctionSpace(N[0], family, bc=bcT, domain=domain[0])
        self.TT = TensorProductSpace(comm, (self.T0, self.F1, self.F2), modify_spaces_inplace=True) # Temperature
        self.uT_ = Function(self.BD)     # Velocity vector times T
        self.T_ = Function(self.TT)      # Temperature solution
        self.Tb = Array(self.TT)

        self.file_T = ShenfunFile('_'.join((filename, 'T')), self.TT, backend='hdf5', mode='w', mesh='uniform')

        # Modify checkpoint file
        self.checkpoint.data['0']['T'] = [self.T_]

        dt = self.dt
        kappa = self.kappa

        # Chebyshev matrices are not sparse, so need a tailored solver. Legendre has simply 5 nonzero diagonals
        sol2 = chebyshev.la.Helmholtz if self.B0.family() == 'chebyshev' else la.SolverGeneric1ND

        # Addition to u equation.
        self.pdes['u'].N = [self.pdes['u'].N, Dx(self.T_, 1, 2)+Dx(self.T_, 2, 2)]
        self.pdes['u'].latex += r'\frac{\partial^2 T}{\partial y^2} + \frac{\partial^2 T}{\partial z^2}'

        # Remove constant pressure gradient from v0 equation
        self.pdes1d['v0'].N = self.pdes1d['v0'].N[0]

        # Add T equation
        q = TestFunction(self.TT)
        self.pdes['T'] = self.PDE(q,
                                  self.T_,
                                  lambda f: kappa*div(grad(f)),
                                  -div(self.uT_),
                                  dt=self.dt,
                                  solver=sol2,
                                  latex=r"\frac{\partial T}{\partial t} = \kappa \nabla^2 T - \nabla \cdot \vec{u}T")

        self.im1 = None
        self.im2 = None

    def update_bc(self, t):
        # Update time-dependent bcs.
        self.T0.bc.update(t)
        self.T_.get_dealiased_space(self.padding_factor).bases[0].bc.update(t)

    def prepare_step(self, rk):
        self.convection()
        Tp = self.T_.backward(padding_factor=self.padding_factor)
        self.uT_ = self.up.function_space().forward(self.up*Tp, self.uT_)

    def tofile(self, tstep):
        self.file_u.write(tstep, {'u': [self.u_.backward(mesh='uniform')]}, as_scalar=True)
        self.file_T.write(tstep, {'T': [self.T_.backward(mesh='uniform')]})

    def init_from_checkpoint(self):
        self.checkpoint.read(self.u_, 'U', step=0)
        self.checkpoint.read(self.T_, 'T', step=0)
        self.g_[:] = 1j*self.K[1]*self.u_[2] - 1j*self.K[2]*self.u_[1]
        self.checkpoint.open()
        tstep = self.checkpoint.f.attrs['tstep']
        t = self.checkpoint.f.attrs['t']
        self.checkpoint.close()
        return t, tstep

    def print_energy_and_divergence(self, t, tstep):
        if tstep % self.moderror == 0 and self.moderror > 0:
            ub = self.u_.backward(self.ub)
            Tb = self.T_.backward(self.Tb)
            e0 = inner(1, ub[0]*ub[0])
            e1 = inner(1, ub[1]*ub[1])
            e2 = inner(1, ub[2]*ub[2])
            d0 = inner(1, Tb*Tb)
            divu = self.divu().backward()
            e3 = np.sqrt(inner(1, divu*divu))
            if comm.Get_rank() == 0:
                if tstep % (10*self.moderror) == 0 or tstep == 0:
                    print(f"{'Time':^11}{'uu':^11}{'vv':^11}{'ww':^11}{'T*T':^11}{'div':^11}")
                print(f"{t:2.4e} {e0:2.4e} {e1:2.4e} {e2:2.4e} {d0:2.4e} {e3:2.4e}")

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
        self.Tb[:] = 0.5*(1-X[0]+0.25*np.sin(np.pi*X[0]))*fun+rand*np.random.randn(*self.Tb.shape)*(1-X[0])*(1+X[0])
        self.T_ = self.Tb.forward(self.T_)
        self.T_.mask_nyquist(self.mask)
        return 0, 0

    def init_plots(self):
        self.ub = ub = self.u_.backward()
        Tb = self.T_.backward(self.Tb)
        if comm.Get_rank() == 0:
            plt.figure(1, figsize=(6, 3))
            #self.im1 = plt.quiver(self.X[1][::4, ::4, 0], self.X[0][::4, ::4, 0], ub[1, ::4, ::4, 0], ub[0, ::4, ::4, 0], pivot='mid', scale=0.01)
            self.im1 = plt.quiver(self.X[1][:, :, 0], self.X[0][:, :, 0], ub[1, :, :, 0], ub[0, :, :, 0], pivot='mid', scale=0.01)
            plt.draw()
            plt.figure(2, figsize=(6, 3))
            self.im2 = plt.contourf(self.X[1][:, :, 0], self.X[0][:, :, 0], Tb[:, :, 0], 100)
            plt.draw()
            plt.pause(1e-6)

    def plot(self, t, tstep):
        if self.im1 is None and self.modplot > 0:
            self.init_plots()

        if tstep % self.modplot == 0 and self.modplot > 0:
            ub = self.u_.backward(self.ub)
            Tb = self.T_.backward(self.Tb)
            if comm.Get_rank() == 0:
                plt.figure(1)
                #self.im1.set_UVC(ub[1, ::4, ::4, 0], ub[0, ::4, ::4, 0])
                self.im1.set_UVC(ub[1, :, :, 0], ub[0, :, :, 0])
                self.im1.scale = np.linalg.norm(ub[1])
                plt.pause(1e-6)
                plt.figure(2)
                self.im2.axes.clear()
                self.im2.axes.contourf(self.X[1][:, :, 0], self.X[0][:, :, 0], Tb[:, :, 0], 100)
                self.im2.autoscale()
                plt.pause(1e-6)

    def solve(self, t=0, tstep=0, end_time=1000):
        c = self.pdes['u'].stages()[2]
        self.assemble()
        while t < end_time-1e-8:
            for rk in range(self.PDE.steps()):
                self.prepare_step(rk)
                for eq in ['u', 'g', 'T']:
                    self.pdes[eq].compute_rhs(rk)
                for eq in ['u', 'g']:
                    self.pdes[eq].solve_step(rk)
                self.compute_vw(rk)
                self.update_bc(t+self.dt*c[rk+1]) # modify time-dep boundary condition
                self.pdes['T'].solve_step(rk)
            t += self.dt
            tstep += 1
            self.update(t, tstep)
            self.checkpoint.update(t, tstep)
            if tstep % self.modsave == 0:
                self.tofile(tstep)

if __name__ == '__main__':
    from time import time
    N = (64, 64, 64)
    d = {
        'N': N,
        'Ra': 1000000.,
        'Pr': 0.7,
        'dt': 0.1,
        'filename': f'RB_{N[0]}_{N[1]}_{N[1]}',
        'conv': 1,
        'modplot': 100,
        'moderror': 10,
        'modsave': 100,
        'bcT': (0.9+0.1*sympy.sin(2*(y-tt)), 0),
        #'bcT': (0.9+0.1*sympy.sin(2*y), 0),
        #'bcT': (1, 0),
        'family': 'C',
        'checkpoint': 100,
        #'padding_factor': 1,
        'timestepper': 'IMEXRK3'
        }
    c = RayleighBenard(**d)
    t, tstep = c.initialize(rand=0.001, from_checkpoint=False)
    t0 = time()
    c.solve(t=t, tstep=tstep, end_time=2)
    print('Computing time %2.4f'%(time()-t0))

