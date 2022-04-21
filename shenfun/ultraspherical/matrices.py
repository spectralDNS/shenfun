import sympy as sp
import numpy as np
from shenfun.matrixbase import SpectralMatrix, SpectralMatDict
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
    def assemble(self):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], Q)
        assert isinstance(trial[0], Q)
        from shenfun.jacobi.recursions import h, n, cn, alfa
        N = test[0].N
        k = np.arange(N, dtype=int)
        a = test[0].alpha
        hh = h(a, a, n, 0, cn)
        d = {0: np.zeros(N)}
        d[0][:] = sp.lambdify(n, hh)(k)
        d[0][0] = sp.simplify(h(alfa, alfa, n, 0, cn).subs(n, 0)).subs(alfa, a) # For Chebyshev, need the correct limit
        return d


mat = SpectralMatDict({
    ((Q,  0), (Q,  0)): BQQmat,
})
