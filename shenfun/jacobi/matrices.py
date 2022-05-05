from shenfun.matrixbase import SpectralMatrix, SpectralMatDict
from shenfun.spectralbase import get_norm_sq
from . import bases

J  = bases.Orthogonal

class BJJmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(J_j, J_k)_w,

    :math:`J_k \in` :class:`.jacobi.bases.Orthogonal` and test and trial spaces have
    dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], J)
        assert isinstance(trial[0], J)
        return {0: get_norm_sq(test[0], trial[0], method)}

mat = SpectralMatDict({
    ((J,  0), (J,  0)): BJJmat,
})
