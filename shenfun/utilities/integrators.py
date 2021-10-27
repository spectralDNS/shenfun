r"""
Module for some integrators.

    - IRK3:     Implicit third order Runge-Kutta
    - RK4:      Runge-Kutta fourth order
    - ETD:      Exponential time differencing Euler method
    - ETDRK4:   Exponential time differencing Runge-Kutta fourth order

See, e.g.,
H. Montanelli and N. Bootland "Solving periodic semilinear PDEs in 1D, 2D and
3D with exponential integrators", https://arxiv.org/pdf/1604.08900.pdf

Integrators are set up to solve equations like

.. math::

    \frac{\partial u}{\partial t} = L u + N(u)

where :math:`u` is the solution, :math:`L` is a linear operator and
:math:`N(u)` is the nonlinear part of the right hand side.

Note
----
`RK4`, `ETD` and `ETDRK4` can only be used with Fourier function spaces,
as they assume all matrices are diagonal.

"""
import types
import numpy as np
from shenfun import Function, TPMatrix, TrialFunction, TestFunction,\
    inner, la, Expr, CompositeSpace, BlockMatrix, extract_bc_matrices,\
    SparseMatrix, get_simplified_tpmatrices, ScipyMatrix

__all__ = ('IRK3', 'BackwardEuler', 'RK4', 'ETDRK4', 'ETD')

#pylint: disable=unused-variable

class IntegratorBase:
    """Abstract base class for integrators

    Parameters
    ----------
        T : TensorProductSpace
        L : function
            To compute linear part of right hand side
        N : function
            To compute nonlinear part of right hand side
        update : function
                 To be called at the end of a timestep
        params : dictionary
                 Any relevant keyword arguments

    """

    def __init__(self, T,
                 L=None,
                 N=None,
                 update=None,
                 **params):
        _p = {'dt': 0}
        _p.update(params)
        self.params = _p
        self.T = T
        if L is not None:
            self.LinearRHS = types.MethodType(L, self)
        if N is not None:
            self.NonlinearRHS = types.MethodType(N, self)
        if update is not None:
            self.update = types.MethodType(update, self)

    def update(self, u, u_hat, t, tstep, **par):
        pass

    def LinearRHS(self, *args, **kwargs):
        return 0

    def NonlinearRHS(self, *args, **kwargs):
        return 0

    def setup(self, dt):
        """Set up solver"""
        pass

    def solve(self, u, u_hat, dt, trange):
        """Integrate forward in time

        Parameters
        ----------
            u : array
                The solution array in physical space
            u_hat : array
                The solution array in spectral space
            dt : float
                Timestep
            trange : two-tuple
                Time and end time
        """
        pass


class IRK3(IntegratorBase):
    """Third order implicit Runge Kutta

    Parameters
    ----------
        T : TensorProductSpace
        L : function of TrialFunction(T)
            To compute linear part of right hand side
        N : function
            To compute nonlinear part of right hand side
        update : function
            To be called at the end of a timestep
        params : dictionary
            Any relevant keyword arguments

    """
    def __init__(self, T,
                 L=None,
                 N=None,
                 update=None,
                 **params):
        IntegratorBase.__init__(self, T, L=L, N=N, update=update, **params)
        self.dU = Function(T)
        self.dU1 = Function(T)
        self.a = (8./15., 5./12., 3./4.)
        self.b = (0.0, -17./60., -5./12.)
        self.c = (0.0, 8./15., 2./3., 1)
        self.solver = None
        self.bm = None
        self.rhs_mats = None
        self.w0 = Function(self.T)
        self.mask = None
        if hasattr(T, 'get_mask_nyquist'):
            self.mask = T.get_mask_nyquist()

    def setup(self, dt):
        if isinstance(self.T, CompositeSpace):
            assert self.T.tensor_rank > 0, 'IRK3 only works for tensors, not generic CompositeSpaces'

        self.params['dt'] = dt
        u = TrialFunction(self.T)
        v = TestFunction(self.T)

        # Note that we are here assembling implicit left hand side matrices,
        # as well as matrices that can be used to assemble the right hand side
        # much faster through matrix-vector products

        a, b = self.a, self.b
        self.solver = []
        self.rhs_mats = []
        u0 = self.LinearRHS(u)
        for rk in range(3):
            if u0:
                mats = inner(v, u-((a[rk]+b[rk])*dt/2)*u0)
            else:
                mats = inner(v, u)
            if self.T.dimensions == 1:
                self.solver.append(la.Solver(mats))

            elif self.T.tensor_rank == 0:
                if len(mats[0].naxes) == 1:
                    self.solver.append(la.SolverGeneric1ND(mats))
                elif len(mats[0].naxes) == 2:
                    self.solver.append(la.SolverGeneric2ND(mats))
                else:
                    raise NotImplementedError
            else:
                self.solver.append(BlockMatrixSolver(mats))

            if u0:
                rhs_mats = inner(v, u+((a[rk]+b[rk])*dt/2)*u0)
            else:
                rhs_mats = inner(v, u)
            mat = ScipyMatrix if self.T.dimensions == 1 else BlockMatrix
            self.rhs_mats.append(mat(rhs_mats))

    def compute_rhs(self, u, u_hat, dU, dU1, rk):
        a = self.a[rk]
        b = self.b[rk]
        dt = self.params['dt']
        dU = self.NonlinearRHS(u, u_hat, dU, **self.params)
        if self.mask:
            dU.mask_nyquist(self.mask)
        w1 = dU*a*dt + dU1*b*dt
        dU1[:] = dU
        if isinstance(dU, np.ndarray):
            dU[:] = w1
        return dU

    def solve(self, u, u_hat, dt, trange):
        if self.solver is None or abs(self.params['dt']-dt) > 1e-12:
            self.setup(dt)
        t, end_time = trange
        tstep = self.tstep = 0
        while t < end_time-1e-8:
            self.tstep = tstep
            for rk in range(3):
                self.params['ti'] = t+self.c[rk]*dt
                dU = self.compute_rhs(u, u_hat, self.dU, self.dU1, rk)
                dU += self.rhs_mats[rk].matvec(u_hat, self.w0)
                u_hat = self.solver[rk](dU, u=u_hat)
                if self.mask:
                    u_hat.mask_nyquist(self.mask)

            t += dt
            tstep += 1
            self.update(u, u_hat, t, tstep, **self.params)

class BackwardEuler(IntegratorBase):
    """First order backward Euler

    Parameters
    ----------
        T : TensorProductSpace
        L : function of TrialFunction(T)
            To compute linear part of right hand side
        N : function
            To compute nonlinear part of right hand side
        update : function
            To be called at the end of a timestep
        params : dictionary
            Any relevant keyword arguments

    """
    def __init__(self, T,
                 L=None,
                 N=None,
                 update=None,
                 **params):
        IntegratorBase.__init__(self, T, L=L, N=N, update=update, **params)
        self.dU = Function(T)
        self.dU1 = Function(T)
        self.solver = None
        self.rhs_mats = None
        self.w0 = Function(self.T)
        self.mask = None
        if hasattr(T, 'get_mask_nyquist'):
            self.mask = T.get_mask_nyquist()

    def setup(self, dt):
        if isinstance(self.T, CompositeSpace):
            assert self.T.tensor_rank > 0, 'BackwardEuler only works for tensors, not generic CompositeSpaces'

        self.params['dt'] = dt
        u = TrialFunction(self.T)
        v = TestFunction(self.T)
        mats = inner(u-dt*self.LinearRHS(u), v)
        M = inner(u, v)

        if self.T.dimensions == 1:
            self.solver = la.Solve(mats)
            self.rhs_mats = M
            return

        if self.T.tensor_rank == 0:
            if len(mats[0].naxes) == 1:
                self.solver = la.SolverGeneric1ND(mats)
            elif len(mats[0].naxes) == 2:
                self.solver = la.SolverGeneric2ND(mats)
            else:
                raise NotImplementedError
        else:
            self.solver = BlockMatrixSolver(mats)
        self.rhs_mats = BlockMatrix(M if isinstance(M, list) else [M])

    def compute_rhs(self, u, u_hat, dU, dU1):
        dt = self.params['dt']
        dU = self.NonlinearRHS(u, u_hat, dU, **self.params)
        if self.mask:
            dU.mask_nyquist(self.mask)
        w1 = dU*2*dt - dU1*dt
        dU1[:] = dU
        return w1

    def solve(self, u, u_hat, dt, trange):
        if self.solver is None or abs(self.params['dt']-dt) > 1e-12:
            self.setup(dt)
        t, end_time = trange
        tstep = self.tstep = 0
        while t < end_time-1e-8:
            self.params['ti'] = t
            self.tstep = tstep
            dU = self.compute_rhs(u, u_hat, self.dU, self.dU1)
            dU += self.rhs_mats.matvec(u_hat, self.w0)
            u_hat = self.solver(dU, u=u_hat)
            if self.mask:
                u_hat.mask_nyquist(self.mask)
            t += dt
            tstep += 1
            self.update(u, u_hat, t, tstep, **self.params)


class ETD(IntegratorBase):
    """Exponential time differencing Euler method

    H. Montanelli and N. Bootland "Solving periodic semilinear PDEs in 1D, 2D and
    3D with exponential integrators", https://arxiv.org/pdf/1604.08900.pdf

    Parameters
    ----------
        T : TensorProductSpace
        L : function
            To compute linear part of right hand side
        N : function
            To compute nonlinear part of right hand side
        update : function
            To be called at the end of a timestep
        params : dictionary
            Any relevant keyword arguments
    """

    def __init__(self, T,
                 L=None,
                 N=None,
                 update=None,
                 **params):
        IntegratorBase.__init__(self, T, L=L, N=N, update=update, **params)
        self.dU = Function(T)
        self.psi = None
        self.ehL = None

    def setup(self, dt):
        """Set up ETD ODE solver"""
        self.params['dt'] = dt
        u = TrialFunction(self.T)
        v = TestFunction(self.T)
        L = self.LinearRHS(u, **self.params)
        if isinstance(L, Expr):
            L = inner(v, L)
            L = get_simplified_tpmatrices(L)[0]
        if isinstance(L, list):
            assert self.T.tensor_rank == 1
            assert L[0].isidentity()
            L = L[0].scale
            # Use only L[0] and let numpy broadcasting take care of the rest
        elif isinstance(L, TPMatrix):
            assert L.isidentity()
            L = L.scale
        elif isinstance(L, SparseMatrix):
            L.simplify_diagonal_matrices()
            L = L.scale

        L = np.atleast_1d(L)
        hL = L*dt
        self.ehL = np.exp(hL)
        M = 50
        psi = self.psi = np.zeros(hL.shape, dtype=float)
        for k in range(1, M+1):
            ll = hL+np.exp(np.pi*1j*(k-0.5)/M)
            psi += ((np.exp(ll)-1.)/ll).real
        psi /= M

    def solve(self, u, u_hat, dt, trange):
        """Integrate forward in time

        Parameters
        ----------
            u : array
                The solution array in physical space
            u_hat : array
                The solution array in spectral space
            dt : float
                Timestep
            trange : two-tuple
                Time and end time
        """
        if self.psi is None or abs(self.params['dt']-dt) > 1e-12:
            self.setup(dt)
        t, end_time = trange
        tstep = 0
        while t < end_time-1e-8:
            t += dt
            tstep += 1
            self.dU = self.NonlinearRHS(u, u_hat, self.dU, **self.params)
            u_hat[:] = self.ehL*u_hat + dt*self.psi*self.dU
            self.update(u, u_hat, t, tstep, **self.params)
        return u_hat


class ETDRK4(IntegratorBase):
    """Exponential time differencing Runge-Kutta 4'th order method

    H. Montanelli and N. Bootland "Solving periodic semilinear PDEs in 1D, 2D and
    3D with exponential integrators", https://arxiv.org/pdf/1604.08900.pdf

    Parameters
    ----------
        T : TensorProductSpace
        L : function
            To compute linear part of right hand side
        N : function
            To compute nonlinear part of right hand side
        update : function
            To be called at the end of a timestep
        params : dictionary
            Any relevant keyword arguments
    """
    def __init__(self, T,
                 L=None,
                 N=None,
                 update=None,
                 **params):
        IntegratorBase.__init__(self, T, L=L, N=N, update=update, **params)
        self.U_hat0 = Function(T)
        self.U_hat1 = Function(T)
        self.dU = Function(T)
        self.dU0 = Function(T)
        self.V2 = Function(T)
        self.psi = np.zeros((4,)+self.U_hat0.shape, dtype=float)
        self.a = None
        self.b = [0.5, 0.5, 0.5]
        self.ehL = None
        self.ehL_h = None

    def setup(self, dt):
        """Set up ETDRK4 ODE solver"""
        self.params['dt'] = dt
        u = TrialFunction(self.T)
        v = TestFunction(self.T)
        L = self.LinearRHS(u, **self.params)
        if isinstance(L, Expr):
            L = inner(v, L)
            L = get_simplified_tpmatrices(L)[0]
        if isinstance(L, list):
            assert self.T.tensor_rank == 1
            assert L[0].isidentity()
            L = L[0].scale
            # Use only L[0] and let numpy broadcasting take care of the rest
        elif isinstance(L, TPMatrix):
            assert L.isidentity()
            L = L.scale
        elif isinstance(L, SparseMatrix):
            L.simplify_diagonal_matrices()
            L = L.scale

        L = np.atleast_1d(L)
        hL = L*dt
        self.ehL = np.exp(hL)
        self.ehL_h = np.exp(hL/2.)
        M = 50
        psi = self.psi = np.zeros((4,) + hL.shape, dtype=float)
        for k in range(1, M+1):
            ll = hL+np.exp(np.pi*1j*(k-0.5)/M)
            psi[0] += ((np.exp(ll)-1.)/ll).real
            psi[1] += ((np.exp(ll)-ll-1.)/ll**2).real
            psi[2] += ((np.exp(ll)-0.5*ll**2-ll-1.)/ll**3).real
            ll2 = hL/2.+np.exp(np.pi*1j*(k-0.5)/M)
            psi[3] += ((np.exp(ll2)-1.)/(ll2)).real

        psi /= M
        a = [psi[0]-3*psi[1]+4*psi[2]]
        a.append(2*psi[1]-4*psi[2])
        a.append(2*psi[1]-4*psi[2])
        a.append(-psi[1]+4*psi[2])
        self.a = a

    def solve(self, u, u_hat, dt, trange):
        """Integrate forward in time

        Parameters
        ----------
            u : array
                The solution array in physical space
            u_hat : array
                The solution array in spectral space
            dt : float
                Timestep
            trange : two-tuple
                Time and end time
        """
        if self.a is None or abs(self.params['dt']-dt) > 1e-12:
            self.setup(dt)
        t, end_time = trange
        tstep = 0
        while t < end_time-1e-8:
            t += dt
            tstep += 1

            self.U_hat0[:] = u_hat*self.ehL_h
            self.U_hat1[:] = u_hat*self.ehL
            for rk in range(4):
                self.dU = self.NonlinearRHS(u, u_hat, self.dU, **self.params)
                if rk < 2:
                    u_hat[:] = self.U_hat0 + self.b[rk]*dt*self.psi[3]*self.dU
                elif rk == 2:
                    u_hat[:] = self.ehL_h*self.V2 + self.b[rk]*dt*self.psi[3]*(2*self.dU-self.dU0)

                if rk == 0:
                    self.dU0[:] = self.dU
                    self.V2[:] = u_hat

                self.U_hat1 += self.a[rk]*dt*self.dU
            u_hat[:] = self.U_hat1
            self.update(u, u_hat, t, tstep, **self.params)
        return u_hat


class RK4(IntegratorBase):
    """Regular 4'th order Runge-Kutta integrator

    Parameters
    ----------
        T : TensorProductSpace
        L : function
            To compute linear part of right hand side
        N : function
            To compute nonlinear part of right hand side
        update : function
            To be called at the end of a timestep
        params : dictionary
            Any relevant keyword arguments
    """
    def __init__(self, T,
                 L=None,
                 N=None,
                 update=None,
                 **params):
        IntegratorBase.__init__(self, T, L=L, N=N, update=update, **params)
        self.U_hat0 = Function(T)
        self.U_hat1 = Function(T)
        self.dU = Function(T)
        self.a = np.array([1./6., 1./3., 1./3., 1./6.])
        self.b = np.array([0.5, 0.5, 1.])

    def setup(self, dt):
        """Set up RK4 ODE solver"""
        self.params['dt'] = dt

    def solve(self, u, u_hat, dt, trange):
        """Integrate forward in end_time

        Parameters
        ----------
            u : array
                The solution array in physical space
            u_hat : array
                The solution array in spectral space
            dt : float
                Timestep
            trange : two-tuple
                Time and end time
        """
        if self.a is None or abs(self.params['dt']-dt) > 1e-12:
            self.setup(dt)
        t, end_time = trange
        tstep = 0
        ut = TrialFunction(self.T)
        vt = TestFunction(self.T)
        L = self.LinearRHS(ut, **self.params)
        if isinstance(L, Expr):
            L = inner(vt, L)
            L = get_simplified_tpmatrices(L)[0]
        if isinstance(L, list):
            assert self.T.tensor_rank == 1
            assert L[0].isidentity()
            L = L[0].scale
            # Use only L[0] and let numpy broadcasting take care of the rest
        elif isinstance(L, TPMatrix):
            assert L.isidentity()
            L = L.scale
        elif isinstance(L, SparseMatrix):
            L.simplify_diagonal_matrices()
            L = L.scale

        while t < end_time-1e-8:
            t += dt
            tstep += 1
            self.U_hat0[:] = self.U_hat1[:] = u_hat
            for rk in range(4):
                dU = self.NonlinearRHS(u, u_hat, self.dU, **self.params)
                if isinstance(L, np.ndarray):
                    dU += L*u_hat
                if rk < 3:
                    u_hat[:] = self.U_hat0 + self.b[rk]*dt*dU
                self.U_hat1 += self.a[rk]*dt*dU
            u_hat[:] = self. U_hat1
            self.update(u, u_hat, t, tstep, **self.params)
        return u_hat
