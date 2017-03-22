from __future__ import division

__all__ = ['mat']

import numpy as np
from shenfun.matrixbase import ShenMatrix
from shenfun.utilities import inheritdocstrings


@inheritdocstrings
class _Fouriermatrix(ShenMatrix):
    def __init__(self, test, trial):
        k = test[0].wavenumbers(test[0].N)
        if isinstance(test[1], (int, np.integer)):
            k_test, k_trial = test[1], trial[1]
        elif isinstance(test[1], np.ndarray):
            assert len(test[1]) == 1
            k_test = test[1][(0,)*np.ndim(test[1])]
            k_trial = trial[1][(0,)*np.ndim(trial[1])]
        else:
            raise RuntimeError

        if k_trial > 0 or k_test > 0:
            val = 2*np.pi*(1j*k)**(k_trial)*(-1j*k)**k_test
            if (k_trial+k_test) % 2 == 0:
                val = val.real
            d = {0: val}
        else:
            d = {0: 2*np.pi}
        ShenMatrix.__init__(self, d, test, trial)

    def solve(self, b, u=None, axis=0):
        N = self.shape[0]
        assert N == b.shape[0]

        if u is None:
            u = b
        else:
            assert u.shape == b.shape

        # Move axis to first
        if axis > 0:
            u = np.moveaxis(u, axis, 0)
            if not u is b:
                b = np.moveaxis(b, axis, 0)

        neglect_zero = False
        if self[0][0] == 0:
            self[0][0] = 1
            neglect_zero = True

        d = 1./self[0]
        sl = [np.newaxis]*u.ndim
        sl[0] = slice(None)
        u[:] = b*d[sl]

        if neglect_zero is True:
            sl[0] = 0
            u[sl] = 0
            self[0][0] = 0

        if axis > 0:
            u = np.moveaxis(u, 0, axis)
            if not u is b:
                b = np.moveaxis(b, axis, 0)

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

