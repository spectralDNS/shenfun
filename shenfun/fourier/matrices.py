"""
Module for handling Fourier diagonal matrices
"""
from __future__ import division

#__all__ = ['mat']

import functools
from numbers import Number
import numpy as np
import sympy as sp
from shenfun.matrixbase import SpectralMatrix
from shenfun import la
from . import bases

R2C = bases.R2C
C2C = bases.C2C

xp = sp.Symbol('x', real=True, positive=True)


class Acos2mat(SpectralMatrix):
    r"""Stiffness matrix for :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj}=(\exp(i k x), \cos^2(x) \exp(i l x)'')

    where test and trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        k = test[0].wavenumbers(bcast=False, scaled=False, eliminate_highest_freq=False)
        N = test[0].N
        d = {0: -0.5*k**2,
             2: -0.25*k[2:]**2,
             -2: -0.25*k[:-2]**2,
             N-2: -0.25*k[-2:]**2,
             -(N-2): -0.25*k[:2]**2}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class Acosmat(SpectralMatrix):
    r"""Stiffness matrix for :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj}=(\exp(i k x), \cos(x) \exp(i l x)'')

    where test and trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        k = test[0].wavenumbers(bcast=False, scaled=False, eliminate_highest_freq=False)
        N = test[0].N
        d = {1: -0.5*k[1:]**2,
             -1: -0.5*k[:-1]**2,
             N-1: -0.5*k[-1]**2}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class Csinmat(SpectralMatrix):
    r"""Derivative matrix for :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(\exp(i k x), \sin(x) \exp(i l x)')

    where test and trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        k = test[0].wavenumbers(bcast=False, scaled=False, eliminate_highest_freq=False)
        N = test[0].N
        d = {1: -0.5*k[1:],
             -1: 0.5*k[:-1],
             N-1: -0.5}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class Csincosmat(SpectralMatrix):
    r"""Derivative matrix for :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(\exp(i k x), \sin(2x)/2 \exp(i l x)')

    where test and trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        k = test[0].wavenumbers(bcast=False, scaled=False, eliminate_highest_freq=False)
        N = test[0].N
        d = {2: -0.25*k[2:],
             -2: 0.25*k[:-2],
             N-2: 0.25*k[-2:],
             -(N-2): -0.25*k[:2]}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class Bcos2mat(SpectralMatrix):
    r"""Matrix for :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\exp(i k x), \cos^2(x) \exp(i l x))

    where test and trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        N = test[0].N
        d = {0: 0.5,
             2: 0.25,
             -2: 0.25,
             N-2: 0.25,
             -(N-2): 0.25}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class Bcosmat(SpectralMatrix):
    r"""Matrix for :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\exp(i k x), \cos(x) \exp(i l x))

    where test and trial spaces have dimensions of M and N, respectively.

    """
    def __init__(self, test, trial, scale=1, measure=1):
        k = test[0].wavenumbers(bcast=False, scaled=False, eliminate_highest_freq=False)
        N = test[0].N
        d = {1: 0.5,
             -1: 0.5,
             N-1: 0.5,
             -(N-1): 0.5}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)


class _Fouriermatrix(SpectralMatrix):
    def __init__(self, test, trial, scale=1, measure=1):
        N = test[0].N
        d = {}
        if isinstance(measure, Number):
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
                d = {0: val*test[0].domain_factor()}
            else:
                d = {0: test[0].domain_factor()}
        SpectralMatrix.__init__(self, d, test, trial, scale=scale, measure=measure)

    def solve(self, b, u=None, axis=0, constraints=()):
        if self.measure == 1:
            N = self.shape[0]

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

        return la.Solver(self)(b, u=u, axis=axis, constraints=constraints)


class _FourierMatDict(dict):
    """Dictionary of inner product matrices.

    Matrices that are missing keys are generated. All Fourier matrices are
    diagonal.

    """

    def __missing__(self, key):
        measure = 1 if len(key) == 2 else key[3]
        c = functools.partial(_Fouriermatrix, measure=measure)
        self[key] = c
        return c

    def __getitem__(self, key):
        matrix = dict.__getitem__(self, key)
        return matrix


mat = _FourierMatDict({
    ((C2C, 0), (C2C, 2), (0, 2*np.pi), sp.cos(xp)**2): functools.partial(Acos2mat, measure=sp.cos(xp)**2),
    ((C2C, 0), (C2C, 2), (0, 2*np.pi), sp.cos(xp)): functools.partial(Acosmat, measure=sp.cos(xp)),
    ((C2C, 0), (C2C, 1), (0, 2*np.pi), sp.sin(xp)): functools.partial(Csinmat, measure=sp.sin(xp)),
    ((C2C, 0), (C2C, 1), (0, 2*np.pi), sp.sin(2*xp)/2): functools.partial(Csincosmat, measure=sp.sin(2*xp)/2),
    ((C2C, 0), (C2C, 1), (0, 2*np.pi), sp.sin(xp)*sp.cos(xp)): functools.partial(Csincosmat, measure=sp.sin(xp)*sp.cos(xp)),
    ((C2C, 0), (C2C, 0), (0, 2*np.pi), sp.cos(xp)**2): functools.partial(Bcos2mat, measure=sp.cos(xp)**2),
    ((C2C, 0), (C2C, 0), (0, 2*np.pi), sp.cos(xp)): functools.partial(Bcosmat, measure=sp.cos(xp)),
})

#mat = _FourierMatDict({})
