from shenfun.matrixbase import SpectralMatrix, SpectralMatDict
from shenfun.la import TDMA_O
from . import bases

CD = bases.CompactDirichlet
CN = bases.CompactNeumann
L = bases.Orthogonal


class BLLmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(La_j, La_k)_w,

    :math:`La_k \in` :class:`.laguerre.bases.Orthogonal` and test and trial spaces have
    dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], L)
        assert isinstance(trial[0], L)
        return {0: 1}

    def solve(self, b, u=None, axis=0, constraints=()):
        if u is not None:
            u[:] = b
            u /= (self.scale*self[0])
            return u

        else:
            b /= (self.scale*self[0])
            return b

    def matvec(self, v, c, format=None, axis=0):
        c[:] = v
        self.scale_array(c, self.scale*self[0])
        return c


class BCDCDmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\phi_j, \phi_k)_w,

    :math:`\phi_k \in` :class:`.laguerre.bases.CompactDirichlet` and test and trial spaces have
    dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], CD)
        assert isinstance(trial[0], CD)
        d = {0:2., 1: -1., -1:-1.}
        return d

    def get_solver(self):
        return TDMA_O


class ACDCDmat(SpectralMatrix):
    r"""Stiffness matrix :math:`BA=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj}=(\phi'_j, \phi'_k)_w,

    :math:`\phi_k \in` :class:`.laguerre.bases.CompactDirichlet` and test and trial spaces have
    dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], CD)
        assert isinstance(trial[0], CD)
        d = {0: 0.5,
             1: 0.25,
             -1: 0.25}
        return d

    def get_solver(self):
        return TDMA_O


mat = SpectralMatDict({
    #((CD, 0), (CD, 0)): BCDCDmat,
    #((CD, 1), (CD, 1)): ACDCDmat,
    ((L, 0), (L, 0)): BLLmat
    })
