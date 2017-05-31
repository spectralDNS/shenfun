import numpy as np
from shenfun.optimization import la
import scipy.linalg as scipy_la
from . import bases

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
            A_scale = A.scale
            B_scale = B.scale

        elif len(args) == 4:
            A = args[0]
            B = args[1]
            A_scale = args[2]
            B_scale = args[3]

        else:
            raise RuntimeError('Wrong input to Helmholtz solver')

        local_shape = A.testfunction[0].forward.input_array.shape
        self.s = A.testfunction[0].slice()
        if local_shape is None:
            M = A[0].shape[0]
            self.d0 = A[0]*A_scale + B[0]*B_scale
            self.d1 = B[2]*B_scale
            self.L = np.zeros_like(self.d1)
            la.TDMA_SymLU(self.d0, self.d1, self.L)

        else:
            self.axis = A.axis
            shape = list(local_shape)
            shape[A.axis] = A.shape[0]
            self.d0 = np.zeros(shape)
            shape[A.axis] = A.shape[0]-2
            self.d1 = np.zeros(shape)
            self.L = np.zeros(shape)

            if len(local_shape) == 2:
                if (isinstance(A.testfunction[0], bases.ShenNeumannBasis) and
                    B_scale[0, 0] == 0):
                    B_scale[0, 0] = 1.
                la.TDMA_SymLU_2D(A, B, A.axis, A_scale[0, 0], B_scale,self. d0,
                                 self.d1, self.L)

            elif len(local_shape) == 3:
                if (isinstance(A.testfunction[0], bases.ShenNeumannBasis) and
                    B_scale[0, 0, 0] == 0):
                    B_scale[0, 0, 0] = 1.

                la.TDMA_SymLU_3D(A, B, A.axis, A_scale[0, 0], B_scale, self.d0,
                                 self.d1, self.L)

            else:
                raise NotImplementedError

    def __call__(self, u, b):
        ss = [slice(None)]*np.ndim(u)
        ss[self.axis] = self.s
        u[ss] = b[ss]
        if u.ndim == 3:
            la.TDMA_SymSolve3D_VC(self.d0, self.d1, self.L, u, self.axis)

        elif u.ndim == 2:
            la.TDMA_SymSolve2D_VC(self.d0, self.d1, self.L, u, self.axis)

        elif u.ndim == 1:
            la.TDMA_SymSolve(self.d0, self.d1, self.L, u)

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

        local_shape = A.testfunction[0].forward.input_array.shape
        if np.ndim(B_scale) > 1:
            self.axis = S.axis
            ss = list(local_shape)
            ss[S.axis] = S[0].shape[0]
            self.d0 = np.zeros(ss)
            ss[S.axis] = A[2].shape[0]
            self.d1 = np.zeros(ss)
            ss[S.axis] = B[4].shape[0]
            self.d2 = np.zeros(ss)
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



class Helmholtz_2dirichlet(object):
    """Helmholtz solver with variable coefficient

    """

    def __init__(self, T, kwargs):

        self.T = T
        if len(kwargs) == 2:
            npaxes = list(kwargs.keys())

            A = kwargs[npaxes[0]]['ADDmat']
            B = kwargs[npaxes[0]]['BDDmat']

            A0 = kwargs[npaxes[1]]['ADDmat']
            B0 = kwargs[npaxes[1]]['BDDmat']

            A_scale = A.scale
            B_scale = B.scale
            A0_scale = A0.scale
            B0_scale = B0.scale

        elif len(args) == 8:
            A = args[0]
            B = args[1]
            A_scale = args[2]
            B_scale = args[3]
            A0 = args[4]
            B0 = args[5]
            A0_scale = args[6]
            B0_scale = args[7]

        else:
            raise RuntimeError('Wrong input to Helmholtz solver')

        # A and A0 are the same matrices if dimensions are equal
        # Assume this to be true for now
        assert A.testfunction[0].N == A0.testfunction[0].N
        assert B.testfunction[0].N == B0.testfunction[0].N

        self.V = np.zeros((A.testfunction[0].N, A.testfunction[0].N))
        self.lmbda = np.ones(A.testfunction[0].N)
        s = self.s = A.testfunction[0].slice()
        self.lmbda[s], self.V[s, s] = scipy_la.eigh(A.diags().toarray(), B.diags().toarray())

        # Create transfer object to realign data in y-direction
        pencilA = T.forward.output_pencil
        pencilB = pencilA.pencil(1)
        self.transAB = pencilA.transfer(pencilB, 'd')
        self.v_hat = np.zeros(self.transAB.subshapeB)

    def __call__(self, u, b):
        s = self.T.local_slice()

        # Map the right hand side to eigen space
        u[:] = (self.V.T).dot(b)
        self.transAB.forward(u, self.v_hat)
        self.v_hat[:] = self.v_hat.dot(self.V)
        self.transAB.backward(self.v_hat, u)

        # Apply the inverse in eigen space
        u /= (self.lmbda[:, np.newaxis] + self.lmbda[np.newaxis, :])[s]

        # Map back to physical space
        u[:] = self.V.dot(u)
        self.transAB.forward(u, self.v_hat)
        self.v_hat[:] = self.v_hat.dot(self.V.T)
        self.transAB.backward(self.v_hat, u)

        return u

