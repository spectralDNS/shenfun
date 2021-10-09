from shenfun import *
import matplotlib.pyplot as plt
import sympy

# pylint: disable=attribute-defined-outside-init

x, y, tt = sympy.symbols('x,y,t', real=True)

class RayleighBenard:

    def __init__(self, N=(32, 32), L=(2, 2*np.pi), Ra=10000., Pr=0.7, dt=0.1,
                 bcT=(0, 1), conv=0, modplot=100, modsave=1e8, filename='RB',
                 family='C', quad='GC'):
        self.nu = np.sqrt(Pr/Ra)
        self.kappa = 1./np.sqrt(Pr*Ra)
        self.dt = dt
        self.N = np.array(N)
        self.L = np.array(L)
        self.conv = conv
        self.modplot = modplot
        self.modsave = modsave
        self.bcT = bcT

        self.a = (8./15., 5./12., 3./4.)
        self.b = (0.0, -17./60., -5./12.)
        self.c = (0., 8./15., 2./3., 1)

        # Regular spaces
        self.sol = chebyshev if family == 'C' else legendre
        self.B0 = FunctionSpace(N[0], family, quad=quad, bc=(0, 0, 0, 0))
        self.D0 = FunctionSpace(N[0], family, quad=quad, bc=(0, 0))
        self.C0 = FunctionSpace(N[0], family, quad=quad)
        self.T0 = FunctionSpace(N[0], family, quad=quad, bc=bcT)
        self.F1 = FunctionSpace(N[1], 'F', dtype='d')
        self.D00 = FunctionSpace(N[0], family, quad=quad, bc=(0, 0))  # Streamwise velocity, not to be in tensorproductspace

        # Regular tensor product spaces
        self.TB = TensorProductSpace(comm, (self.B0, self.F1), modify_spaces_inplace=True) # Wall-normal velocity
        self.TD = TensorProductSpace(comm, (self.D0, self.F1), modify_spaces_inplace=True) # Streamwise velocity
        self.TC = TensorProductSpace(comm, (self.C0, self.F1), modify_spaces_inplace=True) # No bc
        self.TT = TensorProductSpace(comm, (self.T0, self.F1), modify_spaces_inplace=True) # Temperature
        self.BD = VectorSpace([self.TB, self.TD])  # Velocity vector
        self.CD = VectorSpace([self.TD, self.TD])  # Convection vector

        # Padded for dealiasing
        self.TBp = self.TB.get_dealiased((1.5, 1.5))
        self.TDp = self.TD.get_dealiased((1.5, 1.5))
        self.TCp = self.TC.get_dealiased((1.5, 1.5))
        self.TTp = self.TT.get_dealiased((1.5, 1.5))
        #self.TBp = self.TB
        #self.TDp = self.TD
        #self.TCp = self.TC
        #self.TTp = self.TT
        self.BDp = VectorSpace([self.TBp, self.TDp])  # Velocity vector

        self.u_ = Function(self.BD)
        self.ub = Array(self.BD)
        self.up = Array(self.BDp)
        self.w0 = Function(self.TC).v
        self.w1 = Function(self.TC).v
        self.uT_ = Function(self.BD)
        self.T_ = Function(self.TT)
        self.T_b = Array(self.TT)
        self.T_p = Array(self.TTp)
        self.T_1 = Function(self.TT)
        self.rhs_u = Function(self.CD)   # Not important which space, just storage
        self.rhs_T = Function(self.CD)
        self.u00 = Function(self.D00)
        self.b0 = np.zeros((2,)+self.u00.shape)

        self.dudxp = Array(self.TDp)
        self.dudyp = Array(self.TBp)
        self.dvdxp = Array(self.TCp)
        self.dvdyp = Array(self.TDp)

        self.file_u = ShenfunFile('_'.join((filename, 'U')), self.BD, backend='hdf5', mode='w', uniform=True)
        self.file_T = ShenfunFile('_'.join((filename, 'T')), self.TT, backend='hdf5', mode='w', uniform=True)

        self.mask = self.TB.get_mask_nyquist()
        self.K = self.TB.local_wavenumbers(scaled=True)
        self.X = self.TD.local_mesh(True)

        self.H_ = Function(self.CD)  # convection
        self.H_1 = Function(self.CD)
        self.H_2 = Function(self.CD)

        self.curl = Function(self.TCp)
        self.wa = Array(self.TCp)

        self.solver0 = []
        self.B_DD = None
        self.C_DB = None

    def initialize(self, rand=0.001):
        X = self.TB.local_mesh(True)
        funT = 1 if self.bcT[0] == 1 else 2
        fun = {1: 1,
               2: (0.9+0.1*np.sin(2*X[1]))}[funT]
        self.T_b[:] = 0.5*(1-X[0])*fun+rand*np.random.randn(*self.T_b.shape)*(1-X[0])*(1+X[0])
        self.T_ = self.T_b.forward(self.T_)
        self.T_.mask_nyquist(self.mask)
        self.T_1[:] = self.T_

    def assemble(self):
        u = TrialFunction(self.TB)
        v = TestFunction(self.TB)
        p = TrialFunction(self.TT)
        q = TestFunction(self.TT)
        nu = self.nu
        dt = self.dt
        kappa = self.kappa

        # Note that we are here assembling implicit left hand side matrices,
        # as well as matrices that can be used to assemble the right hande side
        # much faster through matrix-vector products

        a, b = self.a, self.b
        self.solver = []
        for rk in range(3):
            mats = inner(v, div(grad(u)) - ((a[rk]+b[rk])*nu*dt/2.)*div(grad(div(grad(u)))))
            self.solver.append(self.sol.la.Biharmonic(mats))

        self.solverT = []
        self.lhs_mat = []
        for rk in range(3):
            matsT = inner(q, 2./(kappa*(a[rk]+b[rk])*dt)*p - div(grad(p)))
            self.lhs_mat.append(extract_bc_matrices([matsT]))
            self.solverT.append(self.sol.la.Helmholtz(matsT))

        u0 = TrialFunction(self.D00)
        v0 = TestFunction(self.D00)
        for rk in range(3):
            mats0 = inner(v0, 2./(nu*(a[rk]+b[rk])*dt)*u0 - div(grad(u0)))
            self.solver0.append(self.sol.la.Helmholtz(mats0))

        self.B_DD = inner(TestFunction(self.TD), TrialFunction(self.TD))
        self.C_DB = inner(Dx(TrialFunction(self.TB), 0, 1), TestFunction(self.TD))

    def end_of_tstep(self):
        self.T_1[:] = self.T_
        self.rhs_T[:] = 0
        self.rhs_u[:] = 0

    def update_bc(self, t):
        # Update the two bases with time-dependent bcs.
        self.T0.bc.update_bcs_time(t)
        self.TTp.bases[0].bc.update_bcs_time(t)

    def compute_curl(self, u):
        return project(Dx(u[1], 0, 1) - Dx(u[0], 1, 1), self.TCp).backward()

    def convection(self, u, H):
        up = self.BDp.backward(u, self.up)
        if self.conv == 0:
            dudxp = project(Dx(u[0], 0, 1), self.TDp).backward(self.dudxp)
            dudyp = project(Dx(u[0], 1, 1), self.TBp).backward(self.dudyp)
            dvdxp = project(Dx(u[1], 0, 1), self.TCp).backward(self.dvdxp)
            dvdyp = project(Dx(u[1], 1, 1), self.TDp).backward(self.dvdyp)
            H[0] = self.TDp.forward(up[0]*dudxp+up[1]*dudyp, H[0])
            H[1] = self.TDp.forward(up[0]*dvdxp+up[1]*dvdyp, H[1])
        elif self.conv == 1:
            curl = self.compute_curl(u)
            H[0] = self.TDp.forward(-curl*up[1])
            H[1] = self.TDp.forward(curl*up[0])
        H.mask_nyquist(self.mask)
        return H

    def compute_rhs_u(self, rhs, rk):
        v = TestFunction(self.TB)
        a = self.a[rk]
        b = self.b[rk]
        H_ = self.convection(self.u_, self.H_)
        rhs[1] = 0
        rhs[1] += inner(v, div(grad(self.u_[0])) + ((a+b)*self.nu*self.dt/2.)*div(grad(div(grad(self.u_[0])))))
        self.w0[:] = 0
        self.w1[:] = 0
        w0 = inner(v, Dx(Dx(H_[1], 0, 1), 1, 1)-Dx(H_[0], 1, 2), output_array=self.w0)
        #w1 = inner(v, div(grad(self.T_)), output_array=self.w1)
        w1 = inner(v, Dx(self.T_, 1, 2), output_array=self.w1)
        rhs[1] += a*self.dt*(w0+w1)
        rhs[1] += b*self.dt*rhs[0]
        rhs[0] = w0+w1
        rhs.mask_nyquist(self.mask)
        return rhs

    def compute_rhs_T(self, rhs, rk):
        a = self.a[rk]
        b = self.b[rk]
        q = TestFunction(self.TT)
        rhs[1] = inner(q, 2./(self.kappa*(a+b)*self.dt)*Expr(self.T_)+div(grad(self.T_)))
        rhs[1] -= self.lhs_mat[rk][0].matvec(self.T_1, self.w0) # T_1 from next step
        up = self.BDp.backward(self.u_, self.up)
        T_p = self.TTp.backward(self.T_, self.T_p)
        uT_ = self.BDp.forward(up*T_p, self.uT_)
        self.w0[:] = 0
        w0 = inner(q, div(uT_), output_array=self.w0)
        rhs[1] -= (2.*a/self.kappa/(a+b))*w0
        rhs[1] -= (2.*b/self.kappa/(a+b))*rhs[0]
        rhs[0] = w0
        rhs.mask_nyquist(self.mask)
        return rhs

    def compute_v(self, u, rk):
        v0 = TestFunction(self.D00)
        if comm.Get_rank() == 0:
            self.u00[:] = u[1, :, 0].real
            w00 = np.zeros_like(self.u00)
        #dudx_hat = project(Dx(u[0], 0, 1), self.TD)
        #with np.errstate(divide='ignore'):
        #    u[1] = 1j * dudx_hat / self.K[1]
        dudx_hat = self.C_DB.matvec(u[0], self.w0)
        with np.errstate(divide='ignore'):
            dudx_hat = 1j * dudx_hat / self.K[1]
        u[1] = self.B_DD.solve(dudx_hat, u=u[1])

        # Still have to compute for wavenumber = 0
        if comm.Get_rank() == 0:
            a, b = self.a[rk], self.b[rk]
            self.b0[1] = inner(v0, 2./(self.nu*(a+b)*self.dt)*Expr(self.u00)+div(grad(self.u00)))
            w00 = inner(v0, self.H_[1, :, 0], output_array=w00)
            self.b0[1] -= (2.*a/self.nu/(a+b))*w00
            self.b0[1] -= (2.*b/self.nu/(a+b))*self.b0[0]
            self.u00 = self.solver0[rk](self.b0[1], self.u00)
            u[1, :, 0] = self.u00
            self.b0[0] = w00
        return u

    def init_plots(self):
        ub = self.u_.backward(self.ub)
        T_b = self.T_.backward(self.T_b)
        if comm.Get_rank() == 0:
            plt.figure(1, figsize=(6, 3))
            self.im1 = plt.quiver(self.X[1], self.X[0], ub[1], ub[0], pivot='mid', scale=0.01)
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
            div_u = project(div(self.u_), self.TD).backward()
            e3 = dx(div_u*div_u)
            if comm.Get_rank() == 0:
                print("Time %2.5f Energy %2.6e %2.6e %2.6e div %2.6e" %(t, e0, e1, e2, e3))

                plt.figure(1)
                self.im1.set_UVC(ub[1], ub[0])
                self.im1.scale = np.linalg.norm(ub[1])
                plt.pause(1e-6)
                plt.figure(2)
                self.im2.ax.clear()
                self.im2.ax.contourf(self.X[1], self.X[0], T_b, 100)
                self.im2.autoscale()
                plt.pause(1e-6)

    def tofile(self, tstep):
        ub = self.u_.backward(self.ub, kind='uniform')
        T_b = self.T_.backward(kind='uniform')
        self.file_u.write(tstep, {'u': [ub]}, as_scalar=True)
        self.file_T.write(tstep, {'T': [T_b]})

    #@profile
    def solve(self, t=0, tstep=0, end_time=1000):
        self.init_plots()
        while t < end_time-1e-8:
            # Fix the new bcs in the solutions. Don't have to fix padded T_p because it is assembled from T_1 and T_2
            for rk in range(3):
                self.T0.bc.set_boundary_dofs(self.T_, True)
                self.update_bc(t+self.dt*self.c[rk+1]) # Update bc for next step
                self.T0.bc.set_boundary_dofs(self.T_1, True) # T_1 holds next step bc
                rhs_u = self.compute_rhs_u(self.rhs_u, rk)
                self.u_[0] = self.solver[rk](rhs_u[1], self.u_[0])
                if comm.Get_rank() == 0:
                    self.u_[0, :, 0] = 0
                u_ = self.compute_v(self.u_, rk)
                u_.mask_nyquist(self.mask)
                rhs_T = self.compute_rhs_T(self.rhs_T, rk)
                T_ = self.solverT[rk](rhs_T[1], self.T_)
                T_.mask_nyquist(self.mask)
                self.T_1 = T_

            t += self.dt
            self.end_of_tstep()
            tstep += 1
            self.plot(t, tstep)
            if tstep % self.modsave == 0:
                self.tofile(tstep)


class RayleighBenard2(RayleighBenard):
    # Faster version of RayleighBenard.

    def assemble(self):
        RayleighBenard.assemble(self)
        u = TrialFunction(self.TB)
        v = TestFunction(self.TB)
        sv = TrialFunction(self.CD)
        p = TrialFunction(self.TT)
        q = TestFunction(self.TT)
        nu = self.nu
        dt = self.dt
        kappa = self.kappa
        a = self.a
        b = self.b

        # Assemble matrices that are used to compute the right hande side
        # through matrix-vector products

        self.mats_u = []
        self.mats_rhs_T = []
        self.rhs_mat = []
        for rk in range(3):
            self.mats_u.append(inner(v, div(grad(u)) + (nu*(a[rk]+b[rk])*dt/2.)*div(grad(div(grad(u))))))
            self.mats_rhs_T.append(inner(q, 2./(kappa*(a[rk]+b[rk])*dt)*p + div(grad(p))))
            self.rhs_mat.append(extract_bc_matrices([self.mats_rhs_T[-1]]))

        #self.mats_uT = inner(v, div(grad(p)))
        self.mats_uT = inner(v, Dx(p, 1, 2))
        self.mat_conv = inner(v, (Dx(Dx(sv[1], 0, 1), 1, 1) - Dx(sv[0], 1, 2)))

        uv = TrialFunction(self.BD)
        self.mats_div_uT = inner(q, div(uv))

        vc = TestFunction(self.TC)
        uc = TrialFunction(self.TC)
        self.A_TC = inner(vc, uc)
        self.curl_rhs = inner(vc, Dx(uv[1], 0, 1) - Dx(uv[0], 1, 1))

        vd = TestFunction(self.TD)
        ud = TrialFunction(self.TD)
        self.A_TD = inner(vd, ud)
        self.CDB = inner(vd, Dx(u, 0, 1))
        self.CTD = inner(vc, Dx(ud, 0, 1))

    def compute_curl(self, u):
        self.w1[:] = 0
        for mat in self.curl_rhs:
            self.w1 += mat.matvec(u, self.w0)
        curl = self.A_TC.solve(self.w1, self.curl)
        curl.mask_nyquist(self.mask)
        return curl.backward(self.wa)

    def convection(self, u, H):
        up = self.BDp.backward(u, self.up)
        if self.conv == 0:
            #dudxp = project(Dx(u[0], 0, 1), self.TDp).backward(self.dudxp)
            self.w0 = self.CDB.matvec(u[0], self.w0)
            self.w0 = self.A_TD.solve(self.w0)
            dudxp = self.TDp.backward(self.w0, self.dudxp)
            #dudyp = project(Dx(u[0], 1, 1), self.TBp).backward(self.dudyp)
            dudyp = self.TBp.backward(1j*self.K[1]*u[0], self.dudyp)
            #dvdxp = project(Dx(u[1], 0, 1), self.TCp).backward(self.dvdxp)
            self.w0 = self.CTD.matvec(u[1], self.w0)
            self.w0 = self.A_TC.solve(self.w0)
            dvdxp = self.TCp.backward(self.w0, self.dvdxp)
            #dvdyp = project(Dx(u[1], 1, 1), self.TDp).backward(self.dvdyp)
            dvdyp = self.TDp.backward(1j*self.K[1]*u[1], self.dvdyp)
            H[0] = self.TDp.forward(up[0]*dudxp+up[1]*dudyp, H[0])
            H[1] = self.TDp.forward(up[0]*dvdxp+up[1]*dvdyp, H[1])
        elif self.conv == 1:
            curl = self.compute_curl(u)
            H[0] = self.TDp.forward(-curl*up[1])
            H[1] = self.TDp.forward(curl*up[0])
        H.mask_nyquist(self.mask)
        return H

    def compute_rhs_u(self, rhs, rk):
        a = self.a[rk]
        b = self.b[rk]
        H_ = self.convection(self.u_, self.H_)
        rhs[1] = 0
        for mat in self.mats_u[rk]:
            rhs[1] += mat.matvec(self.u_[0], self.w0)
        self.w1[:] = 0
        for mat in self.mat_conv:
            self.w1 += mat.matvec(H_, self.w0)
        for mat in self.mats_uT:
            self.w1 += mat.matvec(self.T_, self.w0)
        rhs[1] += a*self.dt*self.w1
        rhs[1] += b*self.dt*rhs[0]
        rhs[0] = self.w1
        rhs.mask_nyquist(self.mask)
        return rhs

    def compute_rhs_T(self, rhs, rk):
        a = self.a[rk]
        b = self.b[rk]
        rhs[1] = 0
        for mat in self.mats_rhs_T[rk]:
            rhs[1] += mat.matvec(self.T_, self.w0) #same as rhs = inner(q, (2./kappa/dt)*Expr(T_1) + div(grad(T_1)), output_array=rhs)

        # The following two are equal as long as the bcs is constant
        # For varying bcs they need to be included
        if isinstance(self.bcT[0], sympy.Expr):
            rhs[1] -= self.lhs_mat[rk][0].matvec(self.T_1, self.w0)
            rhs[1] += self.rhs_mat[rk][0].matvec(self.T_, self.w1)
            #print(np.linalg.norm(self.w0-self.w1))

        up = self.BDp.backward(self.u_, self.up)
        T_p = self.TTp.backward(self.T_, self.T_p)
        uT_ = self.BDp.forward(up*T_p, self.uT_)

        self.w1[:] = 0
        for mat in self.mats_div_uT:
            self.w1 += mat.matvec(uT_, self.w0) # same as rhs -= (2./self.kappa)*inner(q, div(uT_))
        rhs[1] -= (2.*a/self.kappa/(a+b))*self.w1
        rhs[1] -= (2.*b/self.kappa/(a+b))*rhs[0]
        rhs[0] = self.w1
        rhs.mask_nyquist(self.mask)
        return rhs

if __name__ == '__main__':
    from time import time
    from mpi4py_fft import generate_xdmf
    t0 = time()
    d = {
        'N': (64, 128),
        'Ra': 1000000.,
        'dt': 0.005,
        'filename': 'RB100',
        'conv': 1,
        'modplot': 100,
        'modsave': 50,
        #'bcT': (0.9+0.1*sympy.sin(2*(y-tt)), 0),
        'bcT': (1, 0),
        'family': 'C',
        'quad': 'GC'
        }
    c = RayleighBenard2(**d)
    c.initialize(rand=0.001)
    c.assemble()
    c.solve(end_time=200)
    print('Computing time %2.4f'%(time()-t0))
    if comm.Get_rank() == 0:
        generate_xdmf('_'.join((d['filename'], 'U'))+'.h5')
        generate_xdmf('_'.join((d['filename'], 'T'))+'.h5')
