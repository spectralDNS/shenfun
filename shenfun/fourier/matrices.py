from __future__ import division

__all__ = ['mat']

import numpy as np
from shenfun.matrixbase import ShenMatrix
from shenfun.utilities import inheritdocstrings


@inheritdocstrings
class _Fouriermatrix(ShenMatrix):
    def __init__(self, test, trial):
        k = test[0].wavenumbers(test[0].N)
        d = {0: 2*np.pi*(1j*k)**trial[1]}
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

