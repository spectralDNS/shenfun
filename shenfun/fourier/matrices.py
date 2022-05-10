"""
Module for handling Fourier diagonal matrices
"""
from __future__ import division
import functools
from numbers import Number
import numpy as np
import sympy as sp
from shenfun.matrixbase import SpectralMatrix, SpectralMatDict
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
    def assemble(self, method):
        test = self.testfunction
        k = test[0].wavenumbers(bcast=False, scaled=False, eliminate_highest_freq=False)
        N = test[0].N
        d = {0: -0.5*k**2,
             2: -0.25*k[2:]**2,
             -2: -0.25*k[:-2]**2,
             N-2: -0.25*k[-2:]**2,
             -(N-2): -0.25*k[:2]**2}
        return d


class Acosmat(SpectralMatrix):
    r"""Stiffness matrix for :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj}=(\exp(i k x), \cos(x) \exp(i l x)'')

    where test and trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test = self.testfunction
        k = test[0].wavenumbers(bcast=False, scaled=False, eliminate_highest_freq=False)
        N = test[0].N
        d = {1: -0.5*k[1:]**2,
             -1: -0.5*k[:-1]**2,
             N-1: -0.5*k[-1]**2}
        return d


class Csinmat(SpectralMatrix):
    r"""Derivative matrix for :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(\exp(i k x), \sin(x) \exp(i l x)')

    where test and trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test = self.testfunction
        k = test[0].wavenumbers(bcast=False, scaled=False, eliminate_highest_freq=False)
        N = test[0].N
        d = {1: -0.5*k[1:],
             -1: 0.5*k[:-1],
             N-1: -0.5}
        return d


class Csincosmat(SpectralMatrix):
    r"""Derivative matrix for :math:`C=(c_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        c_{kj}=(\exp(i k x), \sin(2x)/2 \exp(i l x)')

    where test and trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test = self.testfunction
        k = test[0].wavenumbers(bcast=False, scaled=False, eliminate_highest_freq=False)
        N = test[0].N
        d = {2: -0.25*k[2:],
             -2: 0.25*k[:-2],
             N-2: 0.25*k[-2:],
             -(N-2): -0.25*k[:2]}
        return d

class Bcos2mat(SpectralMatrix):
    r"""Matrix for :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\exp(i k x), \cos^2(x) \exp(i l x))

    where test and trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test = self.testfunction
        N = test[0].N
        d = {0: 0.5,
             2: 0.25,
             -2: 0.25,
             N-2: 0.25,
             -(N-2): 0.25}
        return d


class Bcosmat(SpectralMatrix):
    r"""Matrix for :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(\exp(i k x), \cos(x) \exp(i l x))

    where test and trial spaces have dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test = self.testfunction
        N = test[0].N
        d = {1: 0.5,
             -1: 0.5,
             N-1: 0.5,
             -(N-1): 0.5}
        return d

class FourierMatDict(SpectralMatDict):
    def __missing__(self, key):
        measure = 1 if len(key) == 2 else key[2]
        c = functools.partial(FourierMatrix, measure=measure)
        self[key] = c
        return c

class FourierMatrix(SpectralMatrix):
    def __init__(self, test, trial, scale=1, measure=1, assemble=None, kind=None, fixed_resolution=None):
        SpectralMatrix.__init__(self, test, trial, scale=scale, measure=measure, assemble=assemble, kind=kind, fixed_resolution=fixed_resolution)

    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        N = test[0].N
        d = {}
        if isinstance(self.measure, Number):
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
                d = {0: val*float(test[0].domain_factor())}

            else:
                d = {0: float(test[0].domain_factor())}
        else:
            d = None
        return d

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


mat = FourierMatDict({
    ((C2C, 0), (C2C, 2), sp.cos(xp)**2): Acos2mat,
    ((C2C, 0), (C2C, 2), sp.cos(xp)): Acosmat,
    ((C2C, 0), (C2C, 1), sp.sin(xp)): Csinmat,
    ((C2C, 0), (C2C, 1), sp.sin(2*xp)/2): Csincosmat,
    ((C2C, 0), (C2C, 1), sp.sin(xp)*sp.cos(xp)): Csincosmat,
    ((C2C, 0), (C2C, 0), sp.cos(xp)**2): Bcos2mat,
    ((C2C, 0), (C2C, 0), sp.cos(xp)): Bcosmat,
})

#mat = FourierMatDict({})
