import numpy as np
from shenfun.matrixbase import SpectralMatrix
from shenfun.utilities import inheritdocstrings


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


mat = _LagMatDict({})
