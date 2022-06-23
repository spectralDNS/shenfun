import matplotlib.pyplot as plt
from shenfun import *
from MicroPolar import MicroPolar
import h5py


class MKM(MicroPolar):

    def __init__(self,
                 N=(32, 32, 32),
                 domain=((-1, 1), (0, 2*np.pi), (0, np.pi)),
                 Re=180.,
                 J=1e-5,
                 m=0.1,
                 NP=8.3e4,
                 dt=0.1,
                 conv=0,
                 modplot=100,
                 modsave=1e8,
                 moderror=100,
                 sample_stats=1e8,
                 filename='MKM_Polar',
                 family='C',
                 padding_factor=(1, 1.5, 1.5),
                 checkpoint=1000,
                 timestepper='IMEXRK3',
                 rand=1e-7):
        MicroPolar.__init__(self, N=N, domain=domain, Re=Re, J=J, m=m, NP=NP, dt=dt, conv=conv, modplot=modplot,
                            modsave=modsave, moderror=moderror, filename=filename, family=family,
                            padding_factor=padding_factor, checkpoint=checkpoint, timestepper=timestepper)
        self.rand = rand
        self.Volume = inner(1, Array(self.TD, val=1))
        self.flux = np.array([618.97]) # Re_tau=180
        self.sample_stats = sample_stats
        self.stats = Stats(N, self.TD.local_slice(False), filename=filename+'_stats')

    def initialize(self, from_checkpoint=False):
        if from_checkpoint:
            return self.init_from_checkpoint()

        X = self.X
        Y = np.where(X[0] < 0, 1+X[0], 1-X[0])
        utau = self.nu*self.Re
        Um = 46.9091*utau # For Re=180
        #Um = 56.*utau
        Xplus = Y*self.Re
        Yplus = X[1]*self.Re
        Zplus = X[2]*self.Re
        duplus = Um*0.2/utau  #Um*0.25/utau
        alfaplus = self.F1.domain[1]/200.  # May have to adjust these two for different Re
        betaplus = self.F2.domain[1]/100.  #
        sigma = 0.00055 # 0.00055
        epsilon = Um/200.   #Um/200.
        U = Array(self.BD)
        U[1] = Um*(Y-0.5*Y**2)
        dev = 1+self.rand*np.random.randn(Y.shape[0], Y.shape[1], Y.shape[2])
        #dev = np.fromfile('dev.dat').reshape((64, 64, 64))
        dd = utau*duplus/2.0*Xplus/40.*np.exp(-sigma*Xplus**2+0.5)*np.cos(betaplus*Zplus)*dev[:, slice(0, 1), :]
        U[1] += dd
        U[2] += epsilon*np.sin(alfaplus*Yplus)*Xplus*np.exp(-sigma*Xplus**2)*dev[:, :, slice(0, 1)]
        u_ = U.forward(self.u_)
        u_.mask_nyquist(self.mask)
        U = u_.backward(U)
        u_ = U.forward(self.u_)
        self.g_[:] = 1j*self.K[1]*u_[2] - 1j*self.K[2]*u_[1]
        return 0, 0

    def init_plots(self):
        ub = self.u_.backward(self.ub)
        self.im1 = 1
        if comm.Get_rank() == 0:

            plt.figure(1, figsize=(6, 3))
            self.im1 = plt.contourf(self.X[1][:, :, 0], self.X[0][:, :, 0], ub[0, :, :, 0], 100)
            plt.colorbar(self.im1)
            plt.draw()

            plt.figure(2, figsize=(6, 3))
            self.im2 = plt.contourf(self.X[1][:, :, 0], self.X[0][:, :, 0], ub[1, :, :, 0], 100)
            plt.colorbar(self.im2)
            plt.draw()

            plt.figure(3, figsize=(6, 3))
            self.im3 = plt.contourf(self.X[2][:, 0, :], self.X[0][:, 0, :], ub[0, :, 0, :], 100)
            plt.colorbar(self.im3)
            plt.draw()

            if comm.Get_size() == 1:
                plt.figure(4, figsize=(6, 3))
                self.im4 = plt.plot(self.X[0][:, 0, 0], ub[0, :, 0, 0])[0]
                plt.draw()


    def plot(self, t, tstep):
        if self.im1 is None and self.modplot > 0:
            self.init_plots()

        if tstep % self.modplot == 0 and self.modplot > 0:
            ub = self.u_.backward(self.ub)
            if comm.Get_rank() == 0:
                X = self.X
                self.im1.axes.clear()
                self.im1.axes.contourf(X[1][:, :, 0], X[0][:, :, 0], ub[0, :, :, 0], 100)
                self.im1.autoscale()
                plt.figure(1)
                plt.pause(1e-6)
                self.im2.axes.clear()
                self.im2.axes.contourf(X[1][:, :, 0], X[0][:, :, 0], ub[1, :, :, 0], 100)
                self.im2.autoscale()
                plt.figure(2)
                plt.pause(1e-6)
                self.im3.axes.clear()
                self.im3.axes.contourf(X[2][:, 0, :], X[0][:, 0, :], ub[0, :, 0, :], 100)
                self.im3.autoscale()
                plt.figure(3)
                plt.pause(1e-6)

    def print_energy_and_divergence(self, t, tstep):
        if tstep % self.moderror == 0 and self.moderror > 0:
            ub = self.u_.backward(self.ub)
            wb = self.w_.backward(self.wb)
            e0 = inner(1, ub[0]*ub[0])
            e1 = inner(1, ub[1]*ub[1])
            e2 = inner(1, ub[2]*ub[2])
            d0 = inner(1, wb[0]*wb[0])
            d1 = inner(1, wb[1]*wb[1])
            d2 = inner(1, wb[2]*wb[2])
            q = inner(1, ub[1])
            divu = self.divu().backward()
            e3 = np.sqrt(inner(1, divu*divu))
            if comm.Get_rank() == 0:
                if tstep % (10*self.moderror) == 0 or tstep == 0:
                    print(f"{'Time':^11}{'uu':^11}{'vv':^11}{'ww':^11}{'a0*a0':^11}{'a1*a1':^11}{'a2*a2':^11}{'flux':^11}{'div':^11}")
                print(f"{t:2.4e} {e0:2.4e} {e1:2.4e} {e2:2.4e} {d0:2.4e} {d1:2.4e} {d2:2.4e} {q:2.4e} {e3:2.4e}")

    def update(self, t, tstep):
        self.plot(t, tstep)
        self.print_energy_and_divergence(t, tstep)
        if tstep % self.sample_stats == 0:
            ub = self.u_.backward(self.ub)
            wb = self.w_.backward(self.wb)
            self.stats(ub, wb)

            if comm.Get_size() == 1:
                stats = self.stats.get_stats()
                u0, w0, uu, ww = stats
                x = c.B0.mesh(bcast=False)
                self.im4.axes.clear()
                self.im4.axes.plot(x, u0[1], 'b')
                self.im4.axes.plot(x, w0[2], 'r')
                plt.figure(4)
                plt.pause(1e-6)

        # Dynamically adjust flux
        if tstep % 1 == 0:
            ub1 = self.u_[1].backward(self.ub[1])
            beta = inner(1, ub1)
            q = (self.flux[0] - beta)
            if comm.Get_rank() == 0:
                self.u_[1, 0, 0, 0] += q/self.Volume

class Stats(object):

    def __init__(self, N, s, fromstats="", filename=""):
        self.N = N # global shape
        self.s = s # local slice
        M = self.s[0].stop-self.s[0].start # local x shape
        self.Q = (self.s[1].stop-self.s[1].start)*(self.s[2].stop-self.s[2].start)
        self.Umean = np.zeros((3, M))
        self.Wmean = np.zeros((3, M))
        self.UU = np.zeros((6, M))
        self.WW = np.zeros((6, M))
        self.num_samples = 0
        self.fname = filename
        self.f0 = None
        if fromstats:
            self.fromfile(filename=fromstats)

    def create_statsfile(self):
        self.f0 = h5py.File(self.fname+".h5", "w", driver="mpio", comm=comm)
        self.f0.create_group("Average Velocity")
        self.f0.create_group("Reynolds Stress Velocity")
        self.f0.create_group("Average Angular Velocity")
        self.f0.create_group("Reynolds Stress Angular Velocity")
        for i in ("U", "V", "W"):
            self.f0["Average Velocity"].create_dataset(i, shape=(self.N[0],), dtype=float)
            self.f0["Average Angular Velocity"].create_dataset(i, shape=(self.N[0],), dtype=float)

        for i in ("UU", "VV", "WW", "UV", "UW", "VW"):
            self.f0["Reynolds Stress Velocity"].create_dataset(i, shape=(self.N[0],), dtype=float)
            self.f0["Reynolds Stress Angular Velocity"].create_dataset(i, shape=(self.N[0],), dtype=float)

    def __call__(self, U, W):
        self.num_samples += 1
        self.Umean += np.sum(U, axis=(2, 3))
        self.Wmean += np.sum(W, axis=(2, 3))
        self.UU[0] += np.sum(U[0]*U[0], axis=(1, 2))
        self.UU[1] += np.sum(U[1]*U[1], axis=(1, 2))
        self.UU[2] += np.sum(U[2]*U[2], axis=(1, 2))
        self.UU[3] += np.sum(U[0]*U[1], axis=(1, 2))
        self.UU[4] += np.sum(U[0]*U[2], axis=(1, 2))
        self.UU[5] += np.sum(U[1]*U[2], axis=(1, 2))
        self.WW[0] += np.sum(W[0]*W[0], axis=(1, 2))
        self.WW[1] += np.sum(W[1]*W[1], axis=(1, 2))
        self.WW[2] += np.sum(W[2]*W[2], axis=(1, 2))
        self.WW[3] += np.sum(W[0]*W[1], axis=(1, 2))
        self.WW[4] += np.sum(W[0]*W[2], axis=(1, 2))
        self.WW[5] += np.sum(W[1]*W[2], axis=(1, 2))
        self.get_stats()

    def get_stats(self, tofile=True):
        s = self.s[0]
        Nd = self.num_samples*self.Q
        comm.barrier()
        if tofile:
            if self.f0 is None:
                self.create_statsfile()
            else:
                self.f0 = h5py.File(self.fname+".h5", "a", driver="mpio", comm=comm)

            for i, name in enumerate(("U", "V", "W")):
                self.f0["Average Velocity/"+name][s] = self.Umean[i]/Nd
                self.f0["Average Angular Velocity/"+name][s] = self.Umean[i]/Nd

            for i, name in enumerate(("UU", "VV", "WW", "UV", "UW", "VW")):
                self.f0["Reynolds Stress Velocity/"+name][s] = self.UU[i]/Nd
                self.f0["Reynolds Stress Angular Velocity/"+name][s] = self.WW[i]/Nd

            self.f0.attrs.create("num_samples", self.num_samples)
            self.f0.close()

        if comm.Get_size() == 1:
            return self.Umean/Nd, self.Wmean/Nd, self.UU/Nd, self.WW/Nd

        if comm.Get_rank() == 0:
            # Return the whole, collected array on rank 0
            f0 = h5py.File(self.fname+".h5", "r", driver='mpio', comm=MPI.COMM_SELF)
            data = (np.array([f0[f'Average Velocity/{name}'] for name in 'UVW']),
                    np.array([f0[f'Average Angular Velocity/{name}'] for name in 'UVW']),
                    np.array([f0[f'Reynolds Stress Velocity/{name}'] for name in ('UU', 'VV', 'WW', 'UV', 'UW', 'VW')]),
                    np.array([f0[f'Reynolds Stress Angular Velocity/{name}'] for name in ('UU', 'VV', 'WW', 'UV', 'UW', 'VW')]))
            f0.close()
            return data

    def reset_stats(self):
        self.num_samples = 0
        self.Umean[:] = 0
        self.Wmean[:] = 0
        self.UU[:] = 0
        self.WW[:] = 0

    def fromfile(self, filename="stats"):
        self.fname = filename
        self.f0 = h5py.File(filename+".h5", "a", driver="mpio", comm=comm)
        self.num_samples = self.f0.attrs["num_samples"]
        M = (self.s[1].stop-self.s[1].start)*(self.s[2].stop-self.s[2].start)
        Nd = self.num_samples*M
        s = self.s[0]
        for i, name in enumerate(("U", "V", "W")):
            self.Umean[i, :] = self.f0["Average Velocity/"+name][s]*Nd
            self.Wmean[i, :] = self.f0["Average Angular Velocity/"+name][s]*Nd
        for i, name in enumerate(("UU", "VV", "WW", "UV", "UW", "VW")):
            self.UU[i, :] = self.f0["Reynolds Stress Velocity/"+name][s]*Nd
            self.WW[i, :] = self.f0["Reynolds Stress Angular Velocity/"+name][s]*Nd
        self.f0.close()

if __name__ == '__main__':
    from time import time
    from mpi4py_fft import generate_xdmf
    t0 = time()
    N = (64, 64, 32)
    d = {
        'N': N,
        'Re': 180.,
        'dt': 0.0005,
        'filename': f'MKM_MP_{N[0]}_{N[1]}_{N[2]}',
        'conv': 1,
        'modplot': 100,
        'modsave': 100,
        'moderror': 1,
        'family': 'C',
        'checkpoint': 1,
        'sample_stats': 10,
        'padding_factor': (1.5, 1.5, 1.5),
        'timestepper': 'IMEXRK222', # IMEXRK222, IMEXRK443
        }
    c = MKM(**d)
    t, tstep = c.initialize(from_checkpoint=False)
    c.solve(t=t, tstep=tstep, end_time=0.01)
    print('Computing time %2.4f'%(time()-t0))
    if comm.Get_rank() == 0:
        generate_xdmf('_'.join((d['filename'], 'U'))+'.h5')
        generate_xdmf('_'.join((d['filename'], 'W'))+'.h5')
    stats = c.stats.get_stats()
    if comm.Get_rank() == 0:
        u0, w0, uu, ww = stats
        x = c.B0.mesh(bcast=False)
        plt.figure()
        plt.semilogx((1-x[:32])*180, u0[1, :32], 'r', (1+x[32:])*180, u0[1, 32:], 'b')
        plt.figure()
        plt.semilogx((1+x[32:])*180, w0[2, 32:], 'b', (1-x[:32])*180, -w0[2, :32], 'r')
        plt.figure()
        # multiply by utau=0.0635 to get the same scaling as fig 7
        plt.semilogx((x[32:] + 1)*180, np.sqrt(ww[1, 32:])*0.0635, 'b', (1 - x[:32])*180, np.sqrt(ww[1, :32])*0.0635, 'r')
        #plt.show()
