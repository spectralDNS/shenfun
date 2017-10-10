import numpy as np
from copy import copy
import scipy.linalg as scipy_la
import scipy.sparse.linalg as sparse_la
from shenfun.optimization import la
from shenfun.la import TDMA as la_TDMA
from . import bases


class TDMA(la_TDMA):

    def __call__(self, b, u=None, axis=0):

        v = self.mat.testfunction[0]
        bc = v.bc

        if u is None:
            u = b
        else:
            assert u.shape == b.shape
            u[:] = b[:]

        if v.is_scaled():
            bc.apply_before(u, False, scales=(-1./np.sqrt(6.), -1./3./np.sqrt(10.)))
        else:
            bc.apply_before(u, False, scales=(-1., -1./3.))

        N = u.shape[axis]
        if not N == self.N:
            self.init(N)

        if len(u.shape) == 3:
            la.TDMA_SymSolve3D(self.dd, self.ud, self.L, u, axis)
            #la.TDMA_SymSolve3D_ptr(self.dd[self.s], self.ud[self.s], self.L,
                                   #u[self.s], axis)
        elif len(u.shape) == 2:
            la.TDMA_SymSolve2D(self.dd, self.ud, self.L, u, axis)

        elif len(u.shape) == 1:
            la.TDMA_SymSolve(self.dd, self.ud, self.L, u)

        else:
            raise NotImplementedError

        bc.apply_after(u, False)

        u /= self.mat.scale
        return u


class Helmholtz(object):
    """Helmholtz solver with variable coefficient

    """

    def __init__(self, *args, **kwargs):

        if 'ADDmat' in kwargs or 'ANNmat' in kwargs:
            if 'ADDmat' in kwargs:
                assert 'BDDmat' in kwargs
                A = kwargs['ADDmat']
                B = kwargs['BDDmat']

            if 'ANNmat' in kwargs:
                assert 'BNNmat' in kwargs
                A = kwargs['ANNmat']
                B = kwargs['BNNmat']
            A_scale = self.A_scale = A.scale
            B_scale = self.B_scale = B.scale

        elif len(args) == 4:
            A = args[0]
            B = args[1]
            A_scale = self.A_scale = args[2]
            B_scale = self.B_scale = args[3]

        else:
            raise RuntimeError('Wrong input to Helmholtz solver')

        self.s = A.testfunction[0].slice()
        neumann = self.neumann = isinstance(A.testfunction[0], bases.ShenNeumannBasis)
        if not neumann:
            self.bc = A.testfunction[0].bc
            self.scaled = A.testfunction[0].is_scaled()

        if np.ndim(B_scale) > 1:
            shape = list(B_scale.shape)
            self.axis = A.axis
            shape[A.axis] = A.shape[0]
            self.d0 = np.zeros(shape)
            shape[A.axis] = A.shape[0]-2
            self.d1 = np.zeros(shape)
            self.L = np.zeros(shape)

            if len(shape) == 2:
                if neumann and B_scale[0, 0] == 0:
                    B_scale[0, 0] = 1.

                if isinstance(A[0], (int, np.integer)):
                    A[0] = np.ones(A.shape[0])

                la.TDMA_SymLU_2D(A, B, A.axis, A_scale[0, 0], B_scale, self.d0,
                                 self.d1, self.L)

            elif len(shape) == 3:
                if neumann and B_scale[0, 0, 0] == 0:
                    B_scale[0, 0, 0] = 1.

                la.TDMA_SymLU_3D(A, B, A.axis, A_scale[0, 0], B_scale, self.d0,
                                 self.d1, self.L)

            else:
                raise NotImplementedError

        else:
            self.d0 = A[0]*A_scale + B[0]*B_scale
            self.d1 = B[2]*B_scale
            self.L = np.zeros_like(self.d1)
            self.bc = A.testfunction[0].bc
            self.axis = 0
            la.TDMA_SymLU(self.d0, self.d1, self.L)


    def __call__(self, u, b):
        ss = [slice(None)]*np.ndim(u)
        ss[self.axis] = self.s
        u[ss] = b[ss]

        #if not self.neumann:
            #s0 = [slice(0, 1)]*u.ndim
            #if self.scaled:
                #u[s0] -= (self.bc[0] + self.bc[1])*self.B_scale[s0]/np.sqrt(6.)
            #else:
                #u[s0] -= (self.bc[0] + self.bc[1])*self.B_scale[s0]
            #s = copy(s0)
            #s[self.axis] = slice(1, 2)
            #if self.scaled:
                #u[s] -= 1./3.*(self.bc[0] - self.bc[1])*self.B_scale[s0]/np.sqrt(10.)
            #else:
                #u[s] -= 1./3.*(self.bc[0] - self.bc[1])*self.B_scale[s0]

        if u.ndim == 3:

            la.TDMA_SymSolve3D_VC(self.d0, self.d1, self.L, u, self.axis)

        elif u.ndim == 2:

            la.TDMA_SymSolve2D_VC(self.d0, self.d1, self.L, u, self.axis)

        elif u.ndim == 1:

            la.TDMA_SymSolve(self.d0, self.d1, self.L, u)

        if not self.neumann:
            self.bc.apply_after(u, True)

        return u


class Biharmonic(object):
    """Biharmonic solver

      a0*u'''' + alfa*u'' + beta*u = b


    """

    def __init__(self, *args, **kwargs):

        if 'SBBmat' in kwargs:
            assert 'PBBmat' in kwargs and 'BBBmat' in kwargs
            S = kwargs['SBBmat']
            A = kwargs['PBBmat']
            B = kwargs['BBBmat']
            S_scale = S.scale
            A_scale = A.scale
            B_scale = B.scale

        elif len(args) == 6:
            S = args[0]
            A = args[1]
            B = args[2]
            S_scale = args[3]
            A_scale = args[4]
            B_scale = args[5]
        else:
            raise RuntimeError('Wrong input to Biharmonic solver')

        if np.ndim(B_scale) > 1:
            shape = list(B_scale.shape)
            self.axis = S.axis
            shape[S.axis] = S[0].shape[0]
            self.d0 = np.zeros(shape)
            shape[S.axis] = A[2].shape[0]
            self.d1 = np.zeros(shape)
            shape[S.axis] = B[4].shape[0]
            self.d2 = np.zeros(shape)
            if np.ndim(B_scale) == 3:
                la.PDMA_SymLU_3D(S, A, B, S.axis, S_scale[0,0,0], A_scale, B_scale, self.d0, self.d1, self.d2)
            elif np.ndim(B_scale) == 2:
                la.PDMA_SymLU_2D(S, A, B, S.axis, S_scale[0,0], A_scale, B_scale, self.d0, self.d1, self.d2)

        else:
            self.d0 = S[0]*S_scale + A[0]*A_scale + B[0]*B_scale
            self.d1 = A[2]*A_scale + B[2]*B_scale
            self.d2 = B[4]*B_scale
            la.PDMA_SymLU(self.d0, self.d1, self.d2)

    def __call__(self, u, b):
        u[:] = b
        if np.ndim(u) == 3:
            la.PDMA_SymSolve3D_VC(self.d0, self.d1, self.d2, u, self.axis)
        elif np.ndim(u) == 2:
            la.PDMA_SymSolve2D_VC(self.d0, self.d1, self.d2, u, self.axis)
        else:
            la.PDMA_Symsolve(self.d0, self.d1, self.d2, u)

        return u

    #def matvec(self, v, c):
        #N = v.shape[0]
        #c[:] = 0
        #if len(v.shape) > 1:
            #Matvec.Biharmonic_matvec3D(v, c, self.a0, self.alfa, self.beta, self.S[0], self.S[2],
                                #self.S[4], self.A[-2], self.A[0], self.A[2],
                                #self.B[-4], self.B[-2], self.B[0], self.B[2], self.B[4])
        #else:
            #Matvec.Biharmonic_matvec(v, c, self.a0, self.alfa, self.beta, self.S[0], self.S[2],
                                #self.S[4], self.A[-2], self.A[0], self.A[2],
                                #self.B[-4], self.B[-2], self.B[0], self.B[2], self.B[4])
        #return c



class Helmholtz_2dirichlet(object): # pragma: no cover
    """Helmholtz solver for 2-dimensional problems with 2 Dirichlet bases.

    a0*BUB + a1*AUB + a2*BUA^T = F

    """

    def __init__(self, T, kwargs):

        self.T = T
        self.V = np.zeros(0)
        assert len(kwargs) == 3

        # There are three terms, BUB, AUB and BUA
        # Extract A and B
        scale = {}
        for tmp in kwargs:
            if tmp[0].get_key() == 'BDDmat' and tmp[1].get_key() == 'BDDmat':
                B = tmp[0]
                B1 = tmp[1]
                scale['BUB'] = tmp['scale']

            elif tmp[0].get_key() == 'ADDmat' and tmp[1].get_key() == 'BDDmat':
                A = tmp[0]
                scale['AUB'] = tmp['scale']

            else:
                A1 = tmp[1]
                scale['BUA'] = tmp['scale']

        # Create transfer object to realign data in second direction
        pencilA = T.forward.output_pencil
        pencilB = pencilA.pencil(1)
        self.pencilB = pencilB
        self.transAB = pencilA.transfer(pencilB, 'd')
        self.u_B = np.zeros(self.transAB.subshapeB)
        self.rhs_A = np.zeros(self.transAB.subshapeA)
        self.rhs_B = np.zeros(self.transAB.subshapeB)

        self.A = A
        self.B = B
        self.A1 = A1
        self.B1 = B1
        self.scale = scale

    def solve_eigen_problem(self, A, B, solver):
        N = A.testfunction[0].N
        s = A.testfunction[0].slice()
        self.V = np.zeros((N, N))
        self.lmbda = np.ones(N)
        if solver == 0:
            self.lmbda[s], self.V[s, s] = scipy_la.eigh(A.diags().toarray(),
                                                        B.diags().toarray())

        elif solver == 1:
            #self.lmbda[s], self.V[s, s] = scipy_la.eigh(B.diags().toarray())
            a = np.zeros((3, N-2))
            a[0, :] = B[0]
            a[2, :-2] = B[2]
            self.lmbda[s], self.V[s, s] = scipy_la.eig_banded(a, lower=True)

    def __call__(self, u, b, solver=1):

        if solver == 0:

            if len(self.V) == 0:
                self.solve_eigen_problem(self.A, self.B, solver)
                self.Vx = self.V
                self.lmbdax = self.lmbda
                if not self.A.testfunction[0].N == self.A1.testfunction[0].N:
                    self.Vx = self.V.copy()
                    self.lmbdax = self.lmbda.copy()
                    self.solve_eigen_problem(self.A1, self.B1, solver)
                    self.Vy = self.V
                    self.lmbday = self.lmbda
                else:
                    self.Vy = self.Vx
                    self.lmbday = self.lmbdax

            # Map the right hand side to eigen space
            u[:] = (self.Vx.T).dot(b)
            self.transAB.forward(u, self.u_B)
            self.u_B[:] = self.u_B.dot(self.Vy)
            self.transAB.backward(self.u_B, u)

            # Apply the inverse in eigen space
            ls = self.T.local_slice()
            u /= (self.scale['BUB'] + self.lmbdax[:, np.newaxis] + self.lmbday[np.newaxis, :])[ls]

            # Map back to physical space
            u[:] = self.Vx.dot(u)
            self.transAB.forward(u, self.u_B)
            self.u_B[:] = self.u_B.dot(self.Vy.T)
            self.transAB.backward(self.u_B, u)

        if solver == 1:

            assert self.A.testfunction[0].is_scaled()

            if len(self.V) == 0:
                self.solve_eigen_problem(self.A, self.B, solver)

            ls = [slice(start, start+shape) for start, shape in zip(self.pencilB.substart,
                                                                    self.pencilB.subshape)]

            self.B1.scale = np.zeros((ls[0].stop-ls[0].start, 1))
            self.B1.scale[:, 0] = self.scale['BUB'] + 1./self.lmbda[ls[0]]
            self.A1.scale = np.ones((1, 1))
            Helmy = Helmholtz(**{'ADDmat': self.A1, 'BDDmat': self.B1})

            # Map the right hand side to eigen space
            self.rhs_A = (self.V.T).dot(b)
            self.rhs_A /= self.lmbda[:, np.newaxis]
            self.transAB.forward(self.rhs_A, self.rhs_B)
            self.u_B = Helmy(self.u_B, self.rhs_B)
            self.transAB.backward(self.u_B, u)
            u[:] = self.V.dot(u)

        elif solver == 2: # iterative
            N = self.A.testfunction[0].N
            s = self.A.testfunction[0].slice()
            AA = np.zeros((N, N))
            BB = np.zeros((N, N))
            G = np.zeros((N, N))
            H = np.zeros((N, N))

            BB[s, s] = self.B.diags().toarray()
            AA[s, s] = self.A.diags().toarray()
            G[:] = BB.dot(u)
            H[:] = u.dot(BB)
            bc = b.copy()
            B_scale = copy(self.B.scale)
            self.B.scale = np.broadcast_to(self.B.scale, (1, u.shape[1])).copy()
            self.B.scale *= self.scale['BUB']
            self.A.scale = np.ones((1, 1))
            Helmx = Helmholtz(**{'ADDmat': self.A, 'BDDmat': self.B})
            converged = False
            G_old = G.copy()
            Hc = H.copy()
            num_iter = 0
            # Solve with successive overrelaxation
            Gc = G.T.copy()
            omega = 1.6
            om = 1.
            while not converged and num_iter < 1000:
                bc[:] = b - G.dot(AA.T)
                Hc = Helmx(Hc, bc)
                H[:] = om*Hc + (1-om)*H[:]
                bc[:] = b.T - (H.T).dot(AA.T)
                Gc = Helmx(Gc, bc)
                G[:] = om*Gc.T + (1-om)*G[:]
                err = np.linalg.norm(G_old-G)
                print('Error ', num_iter, err)
                num_iter += 1
                G_old[:] = G
                converged = err < 1e-8
                om = omega

            self.B.scale = B_scale
            u = self.B.solve(G, u)

        return u
