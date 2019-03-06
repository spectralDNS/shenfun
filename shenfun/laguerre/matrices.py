from shenfun.matrixbase import SpectralMatrix
from shenfun.utilities import inheritdocstrings
from shenfun.la import TDMA_O
from . import bases

LD = bases.ShenDirichletBasis
LB = bases.Basis


@inheritdocstrings
class BLLmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (L_j, L_k)_w

    where

    .. math::

        j = 0, 1, ..., N \text{ and } k = 0, 1, ..., N

    and :math:`L_k` is the Laguerre function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], LB)
        assert isinstance(trial[0], LB)
        SpectralMatrix.__init__(self, {0:1}, test, trial)

    def solve(self, b, u=None, axis=0):
        if u is not None:
            u[:] = b
            u /= self.scale
            return u

        else:
            b /= self.scale
            return b

    def matvec(self, v, c, format='python', axis=0):
        c[:] = v
        self.scale_array(c)
        return c


@inheritdocstrings
class BDDmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        B_{kj} = (\phi_j, \phi_k)_w

    where

    .. math::

        j = 0, 1, ..., N-1 \text{ and } k = 0, 1, ..., N-1

    and :math:`\phi_k` is the Laguerre (function) Dirichlet basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], LD)
        assert isinstance(trial[0], LD)
        d = {0:2., 1: -1., -1:-1.}
        SpectralMatrix.__init__(self, d, test, trial)
        self.solve = TDMA_O(self)

@inheritdocstrings
class ADDmat(SpectralMatrix):
    r"""Mass matrix for inner product

    .. math::

        A_{kj} = (\phi'_j, \phi'_k)_w

    where

    .. math::

        j = 0, 1, ..., N-1 \text{ and } k = 0, 1, ..., N-1

    and :math:`\phi_k` is the Laguerre (function) Dirichlet basis function.

    """
    def __init__(self, test, trial):
        assert isinstance(test[0], LD)
        assert isinstance(trial[0], LD)
        d = {0: 0.5,
             1: 0.25,
             -1: 0.25}
        SpectralMatrix.__init__(self, d, test, trial)
        self.solve = TDMA_O(self)


@inheritdocstrings
class _Lagmatrix(SpectralMatrix):
    def __init__(self, test, trial):
        SpectralMatrix.__init__(self, {}, test, trial)


class _LagMatDict(dict):
    """Dictionary of inner product matrices

    Matrices that are missing keys are generated from Vandermonde type
    computations.

    """

    def __missing__(self, key):
        c = _Lagmatrix
        self[key] = c
        return c

    def __getitem__(self, key):
        matrix = dict.__getitem__(self, key)
        return matrix


mat = _LagMatDict({
    ((LD, 0), (LD, 0)): BDDmat,
    ((LD, 1), (LD, 1)): ADDmat,
    ((LB, 0), (LB, 0)): BLLmat
    })
