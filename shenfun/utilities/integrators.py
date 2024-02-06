r"""
Module for some integrators.

Integrators are set up to solve equations like

.. math::

    \frac{\partial u}{\partial t} = L u + N(u)

where :math:`u` is the solution, :math:`L` is a linear operator and
:math:`N(u)` is the nonlinear part of the right hand side.

There are two kinds of integrators, or time steppers. The first are
ment to be subclassed and used one at the time. These are

    - IRK3:     Implicit third order Runge-Kutta
    - RK4:      Runge-Kutta fourth order
    - ETD:      Exponential time differencing Euler method
    - ETDRK4:   Exponential time differencing Runge-Kutta fourth order

See, e.g.,
H. Montanelli and N. Bootland "Solving periodic semilinear PDEs in 1D, 2D and
3D with exponential integrators", https://arxiv.org/pdf/1604.08900.pdf

The second kind is ment to be used for systems of equations, one class
instance for each equation. These are mainly IMEX Runge Kutta integrators:

    - IMEXRK3
    - IMEXRK111
    - IMEXRK222
    - IMEXRK443

See, e.g., https://github.com/spectralDNS/shenfun/blob/master/demo/ChannelFlow
for an example of use.

Note
----
`RK4`, `ETD` and `ETDRK4` can only be used with Fourier function spaces,
as they assume all matrices are diagonal.

"""
import copy
import types
import numpy as np
from shenfun import Function, TPMatrix, TrialFunction, TestFunction,\
    inner, la, Expr, CompositeSpace, BlockMatrix, SparseMatrix, \
    get_simplified_tpmatrices, ScipyMatrix, Inner, SpectralMatrix

__all__ = ('IRK3', 'BackwardEuler', 'RK4', 'ETDRK4', 'ETD',
           'IMEXRK3', 'IMEXRK011', 'IMEXRK111', 'IMEXRK222', 'IMEXRK443')

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
                self.solver.append(la.BlockMatrixSolver(mats))

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
        if self.mask is not None:
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
                if self.mask is not None:
                    u_hat.mask_nyquist(self.mask)

            t += dt
            tstep += 1
            self.update(u, u_hat, t, tstep, **self.params)
        return u_hat

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
            self.solver = la.BlockMatrixSolver(mats)
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
        return u_hat


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
            if isinstance(L, list):
                if isinstance(L[0], TPMatrix):
                    L = get_simplified_tpmatrices(L)[0]
                elif isinstance(L[0], SpectralMatrix):
                    L = sum(L[1:], L[0])
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


class IMEXRK3:
    r"""Solve partial differential equations of the form

    .. math::

        \frac{\partial u}{\partial t} = N+Lu, \quad (1)

    where :math:`N` is a nonlinear term and :math:`L` is a linear operator.

    This is a third order accurate implicit Runge-Kutta solver

    Parameters
    ----------
    v : :class:`.TestFunction`
    u : :class:`.Expr` or :class:`.Function`
        Representing :math:`u` in (1)
        If an :class:`.Expr`, then its basis must be a :class:`.Function`
        The :class:`.Function` will then hold the solution.
        If :math:`u` is a :class:`.Function`, then this will hold the solution.
    L : Linear operator
        Operates on :math:`u`
    N : :class:`.Expr` or sequence of :class:`.Expr`
        Nonlinear terms
    dt : number
        Time step
    solver : Linear solver, optional
    name : str, optional
    """
    def __init__(self, v, u, L, N, dt, solver=None, name='U-equation', latex=None):
        self.v = v
        self.u = u if isinstance(u, Expr) else Expr(u)
        self.u_ = self.u.basis()
        self.L = L
        self.N = N
        self.dt = dt
        T = self.T = v.function_space()
        self._solver = solver
        if solver is None:
            if v.dimensions == 1:
                self._solver = la.Solver
            elif len(T.get_nondiagonal_axes().ndim) == 1:
                self._solver = la.SolverGeneric1ND
            elif len(T.get_nondiagonal_axes().ndim) == 2:
               self._solver = la.SolverGeneric2ND
            else:
                raise NotImplementedError

        self.solvers = []
        self.linear_rhs = []
        self.nonlinear_rhs = None
        self.name = name
        self.latex = latex
        W = CompositeSpace([T, T])
        self.rhs = Function(W).v
        self.mask = None
        if hasattr(T, 'get_mask_nyquist'):
            self.mask = T.get_mask_nyquist()

    @classmethod
    def steps(cls):
        return 3

    def stages(self):
        a = (8./15., 5./12., 3./4.)
        b = (0.0, -17./60., -5./12.)
        c = (0.0, 8./15., 2./3., 1)
        return a, b, c

    def assemble(self):
        a, b, _ = self.stages()
        dt = self.dt
        ul = copy.copy(self.u)
        ul._basis = TrialFunction(self.u.function_space())
        L1 = self.L(ul)
        L2 = self.L(self.u)
        for rk in range(len(a)):
            mats = inner(self.v, ul - (a[rk]+b[rk])*dt/2*L1)
            self.solvers.append(self._solver(mats))
            self.linear_rhs.append(Inner(self.v, self.u + (a[rk]+b[rk])*dt/2*L2))
        if isinstance(self.N, (Expr, Function)):
            self.nonlinear_rhs = Inner(self.v, self.N)
        elif isinstance(self.N, list):
            self.nonlinear_rhs = sum([Inner(self.v, f) for f in self.N[1:]], start=Inner(self.v, self.N[0]))
        else:
            raise RuntimeError('Wrong type of nonlinear expression')

    def compute_rhs(self, rk=0):
        a, b, _ = self.stages()
        w0 = self.nonlinear_rhs()
        self.rhs[1] = self.dt*(a[rk]*w0+b[rk]*self.rhs[0])
        self.rhs[1] += self.linear_rhs[rk]()
        self.rhs[0] = w0
        if self.mask is not None:
            self.T.mask_nyquist(self.rhs[1], self.mask)
        return self.rhs

    def solve_step(self, rk):
        return self.solvers[rk](self.rhs[-1], self.u_)

class PDEIMEXRK:
    r"""Solve partial differential equations of the form

    .. math::

        \frac{\partial u}{\partial t} = N+Lu, \quad (1)

    where :math:`N` is a nonlinear term and :math:`L` is a linear operator.

    The solvers in this subclass are taken from::

        Ascher, Ruuth and Spiteri 'Implicit-explicit Runge-Kutta methods for
        time-dependent partial differential equations' Applied Numerical
        Mathematics, 25 (1997) 151-167

    Note in particular that we only use solvers satisfying condition
    (2.3).

    Parameters
    ----------
    v : :class:`.TestFunction`
    u : :class:`.Expr` or :class:`.Function`
        Representing :math:`u` in (1)
        If an :class:`.Expr`, then its basis must be a :class:`.Function`
        The :class:`.Function` will then hold the solution.
        If :math:`u` is a :class:`.Function`, then this will hold the solution.
    L : Linear operator
        Operates on :math:`u`
    N : :class:`.Expr` or sequence of :class:`.Expr`
        Nonlinear terms
    dt : number
        Time step
    solver : Linear solver, optional
    name : str, optional
    latex : str, optional
        optional representation of the equation to solve
    """
    def __init__(self, v, u, L, N, dt, solver=None, name='U-equation', latex=None):
        self.v = v
        self.u = u if isinstance(u, Expr) else Expr(u)
        self.u_ = self.u.basis()
        self.L = L
        self.N = N
        self.dt = dt
        self._solver = solver
        if solver is None:
            if v.dimensions > 1:
                self._solver = la.SolverGeneric1ND
            else:
                self._solver = la.Solver

        self.solvers = []
        self.linear_rhs = []
        self.nonlinear_rhs = None
        self.name = name
        self.latex = latex
        T = self.T = v.function_space()
        self.rhs = Function(T).v
        W = CompositeSpace([T]*self.steps())
        self.Krhs = Function(W).v # nonlinear terms
        if self.steps() == 1 and self.Krhs.ndim == 1:
            self.Krhs = np.expand_dims(self.Krhs, 0)

        if self.steps() > 1:
            WL = CompositeSpace([T]*(self.steps()-1))
            self.Lrhs = Function(WL).v # linear terms
            if self.steps() == 2 and self.Lrhs.ndim == 1:
                self.Lrhs = np.expand_dims(self.Lrhs, 0)

        self.mask = None
        if hasattr(T, 'get_mask_nyquist'):
            self.mask = T.get_mask_nyquist()

    @classmethod
    def steps(cls):
        raise NotImplementedError

    def stages(self):
        raise NotImplementedError

    def assemble(self):
        a = self.stages()[0]
        dt = self.dt
        ul = copy.copy(self.u)
        ul._basis = TrialFunction(self.u.function_space())
        mats = inner(self.v, ul - dt*a[1, 1]*self.L(ul))
        self.solvers.append(self._solver(mats))
        self.linear_rhs = Inner(self.v, self.L(self.u))
        self.u0_rhs = Inner(self.v, self.u)
        if isinstance(self.N, (Expr, Function)):
            self.nonlinear_rhs = Inner(self.v, self.N)
        elif isinstance(self.N, list):
            self.nonlinear_rhs = sum([Inner(self.v, f) for f in self.N[1:]], start=Inner(self.v, self.N[0]))
        else:
            raise RuntimeError('Wrong type of nonlinear expression')

    def compute_rhs(self, rk=0):
        a, b = self.stages()[:2]
        self.Krhs[rk] = self.nonlinear_rhs()
        if rk == 0:
            self.u0_rhs() # only at start
        self.rhs[:] = self.u0_rhs.output_array
        for j in range(0, rk+1):
            self.rhs += self.dt*b[rk+1, j]*self.Krhs[j]
        if rk > 0:
            self.Lrhs[rk-1] = self.linear_rhs()
            for j in range(0, rk):
                self.rhs += self.dt*a[rk+1, j+1]*self.Lrhs[j]

        if self.mask is not None:
            self.T.mask_nyquist(self.rhs, self.mask)
        return self.rhs

    def solve_step(self, rk=0):
        # only one solver since the diagonal of a is constant
        return self.solvers[0](self.rhs, self.u_)

class IMEXRK011(PDEIMEXRK):

    @classmethod
    def steps(cls):
        return 1

    def stages(self):
        a = np.array([
            [0, 0],
            [0, 0]])
        b = np.array([
            [0, 0],
            [1, 0]])
        c = (1, 0)
        return a, b, c


class IMEXRK111(PDEIMEXRK):

    @classmethod
    def steps(cls):
        return 1

    def stages(self):
        a = np.array([
            [0, 0],
            [0, 1]])
        b = np.array([
            [0, 0],
            [1, 0]])
        c = (0, 1)
        return a, b, c

class IMEXRK222(PDEIMEXRK):

    @classmethod
    def steps(cls):
        return 2

    def stages(self):
        gamma = (2-np.sqrt(2))/2
        delta = 1-1/(2*gamma)
        a = np.array([
            [0, 0, 0],
            [0, gamma, 0],
            [0, 1-gamma, gamma]])
        b = np.array([
            [0, 0, 0],
            [gamma, 0, 0],
            [delta, 1-delta, 0]])
        c = (0, gamma, 1)
        return a, b, c

class IMEXRK443(PDEIMEXRK):

    @classmethod
    def steps(cls):
        return 4

    def stages(self):
        a = np.array([
            [0, 0, 0, 0, 0],
            [0, 1/2, 0, 0, 0],
            [0, 1/6, 1/2, 0, 0],
            [0, -1/2, 1/2, 1/2, 0],
            [0, 3/2, -3/2, 1/2, 1/2]])
        b = np.array([
            [0, 0, 0, 0, 0],
            [1/2, 0, 0, 0, 0],
            [11/18, 1/18, 0, 0, 0],
            [5/6, -5/6, 1/2, 0, 0],
            [1/4, 7/4, 3/4, -7/4, 0]])
        c = (0, 1/2, 2/3, 1/2, 1)
        return a, b, c
