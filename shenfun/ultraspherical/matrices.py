from shenfun.matrixbase import SpectralMatrix, SpectralMatDict
from shenfun.spectralbase import get_norm_sq
from . import bases

Q  = bases.Orthogonal
CB = bases.CompositeBase
CD = bases.CompactDirichlet
CN = bases.CompactNeumann
UD = bases.UpperDirichlet
LD = bases.LowerDirichlet
P1 = bases.Phi1
P2 = bases.Phi2
P3 = bases.Phi3
P4 = bases.Phi4
BCG = bases.BCGeneric


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


mat = SpectralMatDict({
    ((Q, 0), (Q, 0)): BQQmat,
})
