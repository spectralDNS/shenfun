import numpy as np
from shenfun.matrixbase import SpectralMatrix, SpectralMatDict
from shenfun.optimization import cython
from shenfun.la import TDMA
from . import bases

H = bases.Orthogonal


class BHHmat(SpectralMatrix):
    r"""Mass matrix :math:`B=(b_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        b_{kj}=(H_j, H_k)_w,

    :math:`H_k \in` :class:`.hermite.bases.Orthogonal` and test and trial spaces have
    dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], H)
        assert isinstance(trial[0], H)
        return {0: 1}

    def solve(self, b, u=None, axis=0, constraints=()):
        if u is not None:
            u[:] = b
            u /= (self.scale*self[0])
            return u

        else:
            b /= (self.scale*self[0])
            return b

    def matvec(self, v, c, format=None, axis=0):
        M = self.shape[1]
        ss = [slice(None)]*len(v.shape)
        ss[self.axis] = slice(0, M)
        c[tuple(ss)] = v
        self.scale_array(c, self.scale*self[0])
        return c


class AHHmat(SpectralMatrix):
    r"""Stiffness matrix :math:`A=(a_{kj}) \in \mathbb{R}^{M \times N}`, where

    .. math::

        a_{kj}=(H'_j, H'_k)_w,

    :math:`H_k \in` :class:`.hermite.bases.Orthogonal` and test and trial spaces have
    dimensions of M and N, respectively.

    """
    def assemble(self, method):
        test, trial = self.testfunction, self.trialfunction
        assert isinstance(test[0], H)
        assert isinstance(trial[0], H)
        N = test[0].N
        k = np.arange(N, dtype=float)
        d = {0: k+0.5,
             2: -np.sqrt((k[:-2]+1)*(k[:-2]+2))/2}
        d[0][-1] = (N-1)/2.
        d[-2] = d[2].copy()
        return d

    def get_solver(self):
        return TDMA

    def matvec(self, v, c, format='cython', axis=0):
        N, M = self.shape
        c.fill(0)
        if format == 'cython' and v.ndim == 3:
            ld = self[-2]*np.ones(M-2)
            cython.Matvec.Tridiagonal_matvec3D_ptr(v, c, ld, self[0], ld, axis)
            self.scale_array(c, self.scale)
        elif format == 'cython' and v.ndim == 2:
            ld = self[-2]*np.ones(M-2)
            cython.Matvec.Tridiagonal_matvec2D_ptr(v, c, ld, self[0], ld, axis)
            self.scale_array(c, self.scale)
        elif format == 'cython' and v.ndim == 1:
            ld = self[-2]*np.ones(M-2)
            cython.Matvec.Tridiagonal_matvec(v, c, ld, self[0], ld)
            self.scale_array(c, self.scale)
        elif format == 'self':
            if axis > 0:
                v = np.moveaxis(v, axis, 0)
                c = np.moveaxis(c, axis, 0)
            s = (slice(None),)+(np.newaxis,)*(v.ndim-1) # broadcasting
            c[:(N-2)] = self[2][s]*v[2:N]
            c[:N] += self[0][s]*v[:N]
            c[2:N] += self[-2][s]*v[:(N-2)]
            if axis > 0:
                v = np.moveaxis(v, 0, axis)
                c = np.moveaxis(c, 0, axis)
            self.scale_array(c, self.scale)

        else:
            format = None if format in self._matvec_methods else format
            c = super(AHHmat, self).matvec(v, c, format=format, axis=axis)

        return c


mat = SpectralMatDict({
    ((H, 0), (H, 0)): BHHmat,
    ((H, 1), (H, 1)): AHHmat
    })
