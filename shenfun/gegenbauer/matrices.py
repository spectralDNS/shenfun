import numpy as np
import sympy as sp
from scipy.sparse import spdiags

from shenfun.chebyshev import (
    Orthogonal as CO,
    Phi1, Phi2, Phi4, Phi6
)
from shenfun.matrixbase import SpectralMatDict, SpectralMatrix, extract_diagonal_matrix
from shenfun.spectralbase import get_norm_sq
from shenfun.forms import inner, TestFunction, TrialFunction
from shenfun.jacobi.recursions import h, cn, n
from . import bases

C = bases.Orthogonal
CD = bases.CompactDirichlet
CB = bases.CompactBiharmonic
BCG = bases.BCGeneric

phi = {
    1: Phi1,
    2: Phi2,
    4: Phi4,
    6: Phi6,
}

class BCCmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(C^{(\lambda)}_j, C^{(\lambda)}_k)_w,

    :math:`C^{(\lambda)}_k \in` :class:`.gegenbauer.bases.Orthogonal` and test and
    trial spaces have dimensions of M and N, respectively.

    """

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], C)
        assert isinstance(trial[0], C)
        return {0: get_norm_sq(test[0], trial[0], method)}


class BCTmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(C^{(\lambda)}_j, T_k)_w,

    :math:`C^{(\lambda)}_k \in` :class:`.gegenbauer.bases.Orthogonal` and and
    :math:`T_k \in` :class:`.chebyshev.bases.Chebyshev` test and
    trial spaces have dimensions of M and N, respectively.

    """

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], C)
        assert isinstance(trial[0], CO)
        lambda_ = int(test[0].lambda_)
        test_t = phi[lambda_](test[0].N+2*lambda_)
        A = inner(TestFunction(test_t), TrialFunction(trial[0]), assemble=method)
        hn = h(-sp.S.Half, -sp.S.Half, n+lambda_, lambda_, gn=cn)
        hx = sp.lambdify(n, sp.simplify(hn / (n+lambda_)))(np.arange(test[0].N))
        kk = spdiags([hx / 2**(lambda_-1) / int(sp.factorial(lambda_-1))], [0])
        M = kk @ A.diags()
        keys = np.array(list(A.keys()))
        D = extract_diagonal_matrix(M, lowerband=abs(keys.min()), upperband=keys.max())
        return D._storage

mat = SpectralMatDict(
    {
        ((C, 0), (C, 0)): BCCmat,
        #((C, 0), (CO, 0)): BCTmat,
    }
)
