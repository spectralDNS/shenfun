import numpy as np
from copy import copy
from shenfun.optimization import la, Matvec
from . import bases
from shenfun.la import TDMA as la_TDMA

class TDMA(la_TDMA):

    def __call__(self, b, u=None, axis=0):

        v = self.mat.testfunction[0]
        bc = v.bc

        if u is None:
            u = b
        else:
            assert u.shape == b.shape
            u[:] = b[:]

        bc.apply_before(u, False, scales=(-np.pi/2., -np.pi/4.))

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
    """Helmholtz solver -u'' + alfa*u = b

    args:

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
            A = self.A = args[0]
            B = self.B = args[1]
            A_scale = self.A_scale = args[2]
            B_scale = self.B_scale = args[3]
        else:
            raise RuntimeError('Wrong input to Helmholtz solver')

        #local_shape = A.testfunction[0].forward.input_array.shape
        shape = list(B_scale.shape)
        B[2] = np.broadcast_to(B[2], A[2].shape)
        B[-2] = np.broadcast_to(B[-2], A[2].shape)
        neumann = self.neumann = isinstance(A.testfunction[0], bases.ShenNeumannBasis)
        if not self.neumann:
            self.bc = A.testfunction[0].bc

        if len(shape) == 1:
            N = A.shape[0]+2
            self.u0 = np.zeros(N-2, float)     # Diagonal entries of U
            self.u1 = np.zeros(N-4, float)     # Diagonal+1 entries of U
            self.u2 = np.zeros(N-6, float)     # Diagonal+2 entries of U
            self.L  = np.zeros(N-4, float)     # The single nonzero row of L
            self.axis = 0
            la.LU_Helmholtz_1D(A, B, A_scale, B_scale, neumann, self.u0,
                               self.u1, self.u2, self.L)

        else:
            self.axis = A.axis
            assert A_scale.shape[A.axis] == 1
            assert B_scale.shape[A.axis] == 1

            N = A.shape[0]+2
            A_scale = np.broadcast_to(A_scale, shape).copy()
            shape[A.axis] = N-2
            self.u0 = np.zeros(shape, float)     # Diagonal entries of U
            shape[A.axis] = N-4
            self.u1 = np.zeros(shape, float)     # Diagonal+2 entries of U
            self.L  = np.zeros(shape, float)     # The single nonzero row of L
            shape[A.axis] = N-6
            self.u2 = np.zeros(shape, float)     # Diagonal+4 entries of U
            self.B_scale = B_scale.copy()

            if len(shape) == 2:
                la.LU_Helmholtz_2D(A, B, A.axis, A_scale, B_scale, neumann,
                                   self.u0, self.u1, self.u2, self.L)

            elif len(shape) == 3:
                la.LU_Helmholtz_3D(A, B, A.axis, A_scale, B_scale, neumann,
                                   self.u0, self.u1, self.u2, self.L)

    def __call__(self, u, b):

        # comment since self.B_scale[s0]
        #if not self.neumann:
            #if isinstance(self.bcs, BoundaryValues):
                ##self.bcs.apply_before(b, True)
                #pass

            #else:
                #s0 = [slice(0, 1)]*u.ndim
                #b[s0] -= np.pi/2*(self.bc[0] + self.bc[1])*self.B_scale[s0]
                #s = copy(s0)
                #s[self.axis] = slice(1, 2)
                #b[s] -= np.pi/4*(self.bc[0] - self.bc[1])*self.B_scale[s0]

        if np.ndim(u) == 3:

            #la.Solve_Helmholtz_3D(self.axis, b, u, self.neumann, self.u0, self.u1,
                                  #self.u2, self.L)

            la.Solve_Helmholtz_3D_ptr(self.axis, b, u, self.neumann, self.u0,
                                      self.u1, self.u2, self.L)

            #la.Solve_Helmholtz_3D_hc(self.axis, b, u, self.neumann, self.u0,
                                      #self.u1, self.u2, self.L)


        elif np.ndim(u) == 2:

            la.Solve_Helmholtz_2D(self.axis, b, u, self.neumann, self.u0, self.u1,
                                  self.u2, self.L)

            #la.Solve_Helmholtz_2D_ptr(self.axis, b, u, self.neumann, self.u0,
            #                          self.u1, self.u2, self.L)

        else:
            la.Solve_Helmholtz_1D(b, u, self.neumann, self.u0, self.u1, self.u2, self.L)

        if not self.neumann:
            self.bc.apply_after(u, True)

        return u

    def matvec(self, v, c):
        assert self.neumann is False
        c[:] = 0
        if len(v.shape) > 1:
            raise NotImplementedError
            #Matvec.Helmholtz_matvec3D(v, c, 1.0, self.alfa**2, self.A[0], self.A[2], self.B[0])
        else:
            Matvec.Helmholtz_matvec(v, c, -self.A_scale, self.B_scale, self.A[0], self.A[2], self.B[0])
        return c


class Biharmonic(object):
    """Biharmonic solver

      a0*u'''' + alfa*u'' + beta*u = b


    """

    def __init__(self, *args, **kwargs):

        if 'SBBmat' in kwargs:
            assert 'ABBmat' in kwargs and 'BBBmat' in kwargs
            S = kwargs['SBBmat']
            A = kwargs['ABBmat']
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

        self.S_scale = S_scale
        sii, siu, siuu = S[0], S[2], S[4]
        ail, aii, aiu = A[-2], A[0], A[2]
        bill, bil, bii, biu, biuu = B[-4], B[-2], B[0], B[2], B[4]
        M = sii[::2].shape[0]

        if np.ndim(B_scale) > 1:
            shape = list(B_scale.shape)
            self.axis = S.axis
            shape[S.axis] = M
            ss = copy(shape)
            ss.insert(0, 2)
            S_scale = self.S_scale = np.broadcast_to(S_scale, shape)
            A_scale = np.broadcast_to(A_scale, shape)
            B_scale = np.broadcast_to(B_scale, shape)

            self.u0 = np.zeros(ss)
            self.u1 = np.zeros(ss)
            self.u2 = np.zeros(ss)
            self.l0 = np.zeros(ss)
            self.l1 = np.zeros(ss)
            self.ak = np.zeros(ss)
            self.bk = np.zeros(ss)
            if np.ndim(B_scale) == 3:
                la.LU_Biharmonic_3D_n(S.axis, S_scale, A_scale, B_scale, sii, siu, siuu, ail, aii, aiu, bill, bil, bii, biu, biuu, self.u0, self.u1, self.u2, self.l0, self.l1)
                la.Biharmonic_factor_pr_3D(S.axis, self.ak, self.bk, self.l0, self.l1)

            elif np.ndim(B_scale) == 2:
                la.LU_Biharmonic_2D_n(S.axis, S_scale, A_scale, B_scale, sii, siu, siuu, ail, aii, aiu, bill, bil, bii, biu, biuu, self.u0, self.u1, self.u2, self.l0, self.l1)
                la.Biharmonic_factor_pr_2D(S.axis, self.ak, self.bk, self.l0, self.l1)

        else:
            self.u0 = np.zeros((2, M))
            self.u1 = np.zeros((2, M))
            self.u2 = np.zeros((2, M))
            self.l0 = np.zeros((2, M))
            self.l1 = np.zeros((2, M))
            self.ak = np.zeros((2, M))
            self.bk = np.zeros((2, M))
            la.LU_Biharmonic_1D(S_scale, A_scale, B_scale, sii, siu, siuu, ail, aii, aiu, bill, bil, bii, biu, biuu, self.u0, self.u1, self.u2, self.l0, self.l1)
            la.Biharmonic_factor_pr(self.ak, self.bk, self.l0, self.l1)

    def __call__(self, u, b):
        if np.ndim(u) == 3:
            la.Solve_Biharmonic_3D_n(self.axis, b, u, self.u0, self.u1, self.u2, self.l0, self.l1, self.ak, self.bk, self.S_scale)

        elif np.ndim(u) == 2:
            la.Solve_Biharmonic_2D_n(self.axis, b, u, self.u0, self.u1, self.u2, self.l0, self.l1, self.ak, self.bk, self.S_scale)

        else:
            la.Solve_Biharmonic_1D(b, u, self.u0, self.u1, self.u2, self.l0, self.l1, self.ak, self.bk, self.S_scale)

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


class PDMA(object):
    """Pentadiagonal matrix solver

    Pentadiagonal matrix with diagonals in offsets -4, -2, 0, 2, 4

    kwargs:
        stiffness and mass matrices with scales
        solver      ('cython', 'python')     Choose implementation

    """

    def __init__(self, *args, **kwargs):
        if 'ABBmat' in kwargs:
            assert 'BBBmat' in kwargs
            A = self.A = kwargs['ABBmat']
            B = self.B = kwargs['BBBmat']
            A_scale = self.A_scale = A.scale
            B_scale = self.B_scale = B.scale

        elif len(args) == 4:
            A = self.A = args[0]
            B = self.B = args[1]
            A_scale = self.A_scale = args[2]
            B_scale = self.B_scale = args[3]
        else:
            raise RuntimeError('Wrong input to Helmholtz solver')

        self.solver = kwargs.get('solver', 'cython')
        self.d, self.u1, self.u2 = np.zeros_like(B[0]), np.zeros_like(B[2]), np.zeros_like(B[4])
        self.l1, self.l2 = np.zeros_like(B[2]), np.zeros_like(B[4])
        shape = list(B_scale.shape)

        if len(shape) == 1:
            if self.solver == 'python':
                H = self.A_scale[0]*self.A + self.B_scale[0]*self.B
                self.d[:] = H[0]
                self.u1[:] = H[2]
                self.u2[:] = H[4]
                self.l1[:] = H[-2]
                self.l2[:] = H[-4]
                self.PDMA_LU(self.l2, self.l1, self.d, self.u1, self.u2)

            elif self.solver == 'cython':
                la.LU_Helmholtz_Biharmonic_1D(self.A, self.B,
                        self.A_scale[0], self.B_scale[0],
                        self.l2, self.l1, self.d, self.u1, self.u2)
        else:

            self.axis = A.axis
            assert A_scale.shape[A.axis] == 1
            assert B_scale.shape[A.axis] == 1
            N = A.shape[0]+4
            A_scale = np.broadcast_to(A_scale, shape).copy()
            shape[A.axis] = N-4
            self.d = np.zeros(shape, float)     # Diagonal entries of U
            shape[A.axis] = N-6
            self.u1 = np.zeros(shape, float)     # Diagonal+2 entries of U
            self.l1 = np.zeros(shape, float)     # Diagonal-2 entries of U
            shape[A.axis] = N-8
            self.u2 = np.zeros(shape, float)     # Diagonal+4 entries of U
            self.l2 = np.zeros(shape, float)     # Diagonal-4 entries of U
            self.B_scale = B_scale.copy()

            if len(shape) == 2:
                raise NotImplementedError

            elif len(shape) == 3:
                la.LU_Helmholtz_Biharmonic_3D(A, B, A.axis, A_scale, B_scale,
                                   self.l2, self.l1, self.d, self.u1, self.u2)


    def PDMA_LU(self, l2, l1, d, u1, u2):
        n = d.shape[0]
        m = u1.shape[0]
        k = n - m

        for i in range(n-2*k):
            lam = l1[i]/d[i]
            d[i+k] -= lam*u1[i]
            u1[i+k] -= lam*u2[i]
            l1[i] = lam
            lam = l2[i]/d[i]
            l1[i+k] -= lam*u1[i]
            d[i+2*k] -= lam*u2[i]
            l2[i] = lam

        i = n-4
        lam = l1[i]/d[i]
        d[i+k] -= lam*u1[i]
        l1[i] = lam
        i = n-3
        lam = l1[i]/d[i]
        d[i+k] -= lam*u1[i]
        l1[i] = lam


    def PDMA_Solve(self, l2, l1, d, u1, u2, b):
        n = d.shape[0]
        bc = np.zeros_like(b)
        bc[:] = b[:]

        bc[2] -= l1[0]*bc[0]
        bc[3] -= l1[1]*bc[1]
        for k in range(4, n):
            bc[k] -= (l1[k-2]*bc[k-2] + l2[k-4]*bc[k-4])

        bc[n-1] /= d[n-1]
        bc[n-2] /= d[n-2]
        bc[n-3] /= d[n-3]
        bc[n-3] -= u1[n-3]*bc[n-1]/d[n-3]
        bc[n-4] /= d[n-4]
        bc[n-4] -= u1[n-4]*bc[n-2]/d[n-4]
        for k in range(n-5,-1,-1):
            bc[k] /= d[k]
            bc[k] -= (u1[k]*bc[k+2]/d[k] + u2[k]*bc[k+4]/d[k])
        b[:] = bc

    def __call__(self, u, b):

        if np.ndim(u) == 3:

            la.Solve_Helmholtz_Biharmonic_3D_ptr(self.A.axis, b, u, self.l2,
                                                 self.l1, self.d, self.u1, self.u2)


        else:

            if self.solver == 'python':
                u[:] = b
                self.PDMA_Solve(self.l2, self.l1, self.d, self.u1, self.u2, u)

            elif self.solver == 'cython':
                if u is b:
                    u = np.zeros_like(b)
                #la.Solve_Helmholtz_Biharmonic_1D(b, u, self.l2, self.l1, self.d, self.u1, self.u2)
                la.Solve_Helmholtz_Biharmonic_1D_p(b, u, self.l2, self.l1, self.d, self.u1, self.u2)

        #u /= self.mat.scale
        return u
