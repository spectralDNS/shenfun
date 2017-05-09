import numpy as np
from shenfun.optimization import la, Matvec
from . import bases

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
            A_scale = A.scale
            B_scale = B.scale

        elif len(args) == 4:
            A = self.A = args[0]
            B = self.B = args[1]
            A_scale = self.A_scale = args[2]
            B_scale = self.B_scale = args[3]
        else:
            raise RuntimeError('Wrong input to Helmholtz solver')

        local_shape = A.testfunction[0].forward.input_array.shape
        B[2] = np.broadcast_to(B[2], A[2].shape)
        B[-2] = np.broadcast_to(B[-2], A[2].shape)
        neumann = self.neumann = isinstance(A.testfunction[0], bases.ShenNeumannBasis)
        if local_shape is None:
            N = A.shape[0]+2
            self.u0 = np.zeros(N-2, float)     # Diagonal entries of U
            self.u1 = np.zeros(N-4, float)     # Diagonal+1 entries of U
            self.u2 = np.zeros(N-6, float)     # Diagonal+2 entries of U
            self.L  = np.zeros(N-4, float)     # The single nonzero row of L
            la.LU_Helmholtz_1D(A, B, A_scale, B_scale, neumann, self.u0,
                               self.u1, self.u2, self.L)

        else:
            self.axis = A.axis
            assert A_scale.shape[A.axis] == 1
            assert B_scale.shape[A.axis] == 1

            N = A.shape[0]+2
            A_scale = np.broadcast_to(A_scale, B_scale.shape)
            shape = list(local_shape)
            shape[A.axis] = N-2
            self.u0 = np.zeros(shape, float)     # Diagonal entries of U
            shape[A.axis] = N-4
            self.u1 = np.zeros(shape, float)     # Diagonal+2 entries of U
            self.L  = np.zeros(shape, float)     # The single nonzero row of L
            shape[A.axis] = N-6
            self.u2 = np.zeros(shape, float)     # Diagonal+4 entries of U

            if len(local_shape) == 2:
                la.LU_Helmholtz_2D(A, B, A.axis, A_scale, B_scale, neumann,
                                   self.u0, self.u1, self.u2, self.L)

            elif len(local_shape) == 3:
                la.LU_Helmholtz_3D(A, B, A.axis, A_scale, B_scale, neumann,
                                   self.u0, self.u1, self.u2, self.L)

    def __call__(self, u, b):
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

    args:
        N            integer        Size of problem in real space
        a0           float          Coefficient
        alfa, beta float/arrays     Coefficients. Just one value for 1D problems
                                    and 2D arrays for 3D problems.
    kwargs:
        quad        ('GL', 'GC')    Chebyshev-Gauss-Lobatto or Chebyshev-Gauss
        solver ('cython', 'python') Choose implementation

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

        local_shape = A.testfunction[0].forward.input_array.shape
        self.S_scale = S_scale
        sii, siu, siuu = S[0], S[2], S[4]
        ail, aii, aiu = A[-2], A[0], A[2]
        bill, bil, bii, biu, biuu = B[-4], B[-2], B[0], B[2], B[4]
        M = sii[::2].shape[0]

        if np.ndim(B_scale) > 1:
            self.axis = S.axis
            ss = list(local_shape)
            ss[S.axis] = M
            ss.insert(0, 2)
            S_scale = self.S_scale = np.broadcast_to(S_scale, local_shape)
            A_scale = np.broadcast_to(A_scale, local_shape)
            B_scale = np.broadcast_to(B_scale, local_shape)

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

