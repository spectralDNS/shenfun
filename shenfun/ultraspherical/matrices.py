from shenfun.matrixbase import SpectralMatDict, SpectralMatrix
from shenfun.spectralbase import get_norm_sq

from .bases import Orthogonal

class BQQmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(Q_j, Q_k)_w,

    :math:`Q_k \in` :class:`.ultraspherical.bases.Orthogonal` and test and
    trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], Q)
        assert isinstance(trial[0], Q)
        return {0: get_norm_sq(test[0], trial[0], method)}

Q = Orthogonal

mat = SpectralMatDict({
    ((Q, 0), (Q, 0)): BQQmat,
})
