"""
Module for handling Fourier diagonal matrices
"""
from __future__ import division

__all__ = ['mat']

import numpy as np
from shenfun.matrixbase import SpectralMatrix
from shenfun.utilities import inheritdocstrings

@inheritdocstrings
class _Fouriermatrix(SpectralMatrix):
    def __init__(self, test, trial):
        N = test[0].N
        k = test[0].wavenumbers(N, scaled=False)
        if isinstance(test[1], (int, np.integer)):
            k_test, k_trial = test[1], trial[1]
        elif isinstance(test[1], np.ndarray):
            assert len(test[1]) == 1
            k_test = test[1][(0,)*np.ndim(test[1])]
            k_trial = trial[1][(0,)*np.ndim(trial[1])]
        else:
            raise RuntimeError

        if abs(k_trial) + abs(k_test) > 0:
            if N % 2 == 0 and (k_trial + k_test) % 2 == 1:
                pass
                #k[N//2] = 0
            val = (1j*k)**(k_trial)*(-1j*k)**k_test
            if (k_trial + k_test) % 2 == 0:
                val = val.real
            d = {0: val}
        else:
            d = {0: 1.0}
        SpectralMatrix.__init__(self, d, test, trial)

    def solve(self, b, u=None, axis=0):
        N = self.shape[0]
        assert N == b.shape[axis]

        if u is None:
            u = b
        else:
            assert u.shape == b.shape

        with np.errstate(divide='ignore'):
            d = 1./self[0]
        if isinstance(d, np.ndarray):
            if np.isinf(d[0]):
                d[0] = 0
            if np.isinf(d[N//2]):
                d[N//2] = 0
            sl = [np.newaxis]*u.ndim
            sl[axis] = slice(None)
            u[:] = b*d[tuple(sl)]
        else:
            u[:] = b*d

        u /= self.scale
        return u


class _FourierMatDict(dict):
    """Dictionary of inner product matrices.

    Matrices that are missing keys are generated. All Fourier matrices are
    diagonal.

    """

    def __missing__(self, key):
        c = _Fouriermatrix
        self[key] = c
        return c

    def __getitem__(self, key):
        matrix = dict.__getitem__(self, key)
        return matrix


mat = _FourierMatDict({})
