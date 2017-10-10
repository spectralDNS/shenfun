from __future__ import division

__all__ = ['mat']

import numpy as np
from shenfun.matrixbase import SpectralMatrix
from shenfun.utilities import inheritdocstrings
from .bases import R2CBasis, C2CBasis

@inheritdocstrings
class _Fouriermatrix(SpectralMatrix):
    def __init__(self, test, trial):
        k = test[0].wavenumbers(test[0].N, scaled=False)
        if isinstance(test[1], (int, np.integer)):
            k_test, k_trial = test[1], trial[1]
        elif isinstance(test[1], np.ndarray):
            assert len(test[1]) == 1
            k_test = test[1][(0,)*np.ndim(test[1])]
            k_trial = trial[1][(0,)*np.ndim(trial[1])]
        else:
            raise RuntimeError

        if k_trial > 0 or k_test > 0:
            #if test[0].N % 2 == 0:
                #k[test[0].N//2] = 0
            val = 2*np.pi*(1j*k)**(k_trial)*(-1j*k)**k_test
            if (k_trial+k_test) % 2 == 0:
                val = val.real
            d = {0: val}
        else:
            d = {0: 2*np.pi}
        SpectralMatrix.__init__(self, d, test, trial)

    def solve(self, b, u=None, axis=0, neglect_zero_wavenumber=True):
        N = self.shape[0]
        assert N == b.shape[0]

        if u is None:
            u = b
        else:
            assert u.shape == b.shape

        start = 0
        if neglect_zero_wavenumber is True:
            start = 1

        sl = [np.newaxis]*u.ndim
        sl[axis] = slice(start, None)
        su = [slice(None)]*u.ndim
        su[axis] = slice(start, None)
        d = np.zeros_like(self[0])
        d[slice(start, None)] = 1./self[0][slice(start, None)]
        u[su] = b[su]*d[sl]

        if neglect_zero_wavenumber:
            su[axis] = 0
            u[su] = 0

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

