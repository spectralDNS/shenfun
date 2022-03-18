#pylint: disable=line-too-long, len-as-condition, missing-docstring, too-many-instance-attributes

import numpy as np
import scipy.linalg as scipy_la
from mpi4py import MPI

comm = MPI.COMM_WORLD

class Helmholtz_2dirichlet:
    """Helmholtz solver for 2-dimensional problems with 2 Dirichlet bases.

    .. math::

        a_0 BUB + a_1 AUB + a_2 BUA^T = F

    Somewhat experimental.

    """

    def __init__(self, matrices):

        self.V = np.zeros(0)
        self.matrices = matrices
        assert len(matrices) == 3

        # There are three terms, BUB, AUB and BUA
        # Extract A and B
        scale = {}
        for tmp in matrices:
            mats = tmp.mats
            if mats[0].get_key() == 'BSDSDmat' and mats[1].get_key() == 'BSDSDmat':
                B = mats[0]
                B1 = mats[1]
                scale['BUB'] = tmp.scale
                self.BB = tmp

            elif mats[0].get_key() == 'ASDSDmat' and mats[1].get_key() == 'BSDSDmat':
                A = mats[0]
                scale['AUB'] = tmp.scale
                self.AB = tmp

            else:
                A1 = mats[1]
                scale['BUA'] = tmp.scale
                self.BA = tmp

        # Create transfer object to realign data in second direction
        self.T = T = matrices[0].space
        pencilA = T.forward.output_pencil
        pencilB = T.forward.input_pencil
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

        self.lmbda = None
        self.lmbdax = None
        self.lmbday = None
        self.Vx = None
        self.Vy = None

    def solve_eigen_problem(self, A, B, solver):
        """Solve the eigen problem"""
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

    def __call__(self, b, u, solver=1):
        from shenfun import FunctionSpace, TensorProductSpace, TPMatrix, Identity, la

        if solver == 0: # pragma: no cover

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
            u /= (self.BB.scale + self.lmbdax[:, np.newaxis] + self.lmbday[np.newaxis, :])[ls]

            # Map back to physical space
            u[:] = self.Vx.dot(u)
            self.transAB.forward(u, self.u_B)
            self.u_B[:] = self.u_B.dot(self.Vy.T)
            self.transAB.backward(self.u_B, u)

        elif solver == 1:
            #FIXME Should improve this after redesign of solvers

            if comm.Get_size() > 1:
                raise RuntimeError('Currently broken for MPI > 1')

            assert self.A.testfunction[0].is_scaled()

            if len(self.V) == 0:
                self.solve_eigen_problem(self.A, self.B, solver)

            ls = [slice(start, start+shape) for start, shape in zip(self.pencilB.substart,
                                                                    self.pencilB.subshape)]

            B1_scale = np.zeros((ls[0].stop-ls[0].start, 1))
            B1_scale[:, 0] = self.BB.scale + 1./self.lmbda[ls[0]]
            A1_scale = self.scale['AUB']
            # Create Helmholtz solver along axis=1
            N0 = self.matrices[0].mats[0].shape[0]

            F = FunctionSpace(N0, 'L', dtype='d')
            FT = TensorProductSpace(comm, (F, self.T.bases[1].get_unplanned()))
            AA = TPMatrix([Identity((N0, N0)), self.A1], FT, self.T, A1_scale)
            BB = TPMatrix([Identity((N0, N0)), self.B1], FT, self.T, B1_scale)
            AA._issimplified = True
            BB._issimplified = True
            AA.naxes = [1]
            BB.naxes = [1]

            Helmy = la.SolverGeneric1ND([AA, BB])
            # Map the right hand side to eigen space
            self.rhs_A = self.V.T.dot(b)
            self.rhs_A /= self.lmbda[:, np.newaxis]
            self.transAB.forward(self.rhs_A, self.rhs_B)
            self.u_B = Helmy(self.rhs_B, self.u_B, fast=False)
            self.transAB.backward(self.u_B, u)
            u[:] = self.V.dot(u)

        return u
