#pylint: disable=missing-docstring

import numpy as np
from .bases import *
from .matrices import *

def energy_fourier(u, T):
    r"""Compute the energy of u using Parceval's theorem

    .. math::

        \int abs(u)^2 dx = N*\sum abs(u_hat)^2

    Parameters
    ----------
        u : Array
            The Fourier coefficients
        T : TensorProductSpace

    See https://en.wikipedia.org/wiki/Parseval's_theorem

    """

    if not hasattr(T, 'comm'):
        # Just a 1D basis
        assert u.ndim == 1
        if isinstance(T, R2C):
            if u.shape[0] % 2 == 0:
                result = (2*np.sum(abs(u[1:-1])**2) +
                          np.sum(abs(u[0])**2) +
                          np.sum(abs(u[-1])**2))
            else:
                result = (2*np.sum(abs(u[1:])**2) +
                          np.sum(abs(u[0])**2))

        else:
            result = np.sum(abs(u)**2)
        return result

    comm = T.comm
    assert np.all([base.family() == 'fourier' for base in T.bases])
    real = False
    for axis, base in enumerate(T.bases):
        if isinstance(base, R2C):
            real = True
            break

    if real:
        s = [slice(None)]*u.ndim
        uaxis = axis + u.ndim-len(T.bases)
        if T.forward.output_pencil.subcomm[axis].Get_size() == 1:
            # aligned in r2c direction
            if base.N % 2 == 0:
                s[uaxis] = slice(1, -1)
                result = 2*np.sum(abs(u[tuple(s)])**2)
                s[uaxis] = 0
                result += np.sum(abs(u[tuple(s)])**2)
                s[uaxis] = -1
                result += np.sum(abs(u[tuple(s)])**2)
            else:
                s[uaxis] = slice(1, None)
                result = 2*np.sum(abs(u[tuple(s)])**2)
                s[uaxis] = 0
                result += np.sum(abs(u[tuple(s)])**2)

        else:
            # Data not aligned along r2c axis. Need to check about 0 and -1
            if base.N % 2 == 0:
                s[uaxis] = slice(1, -1)
                result = 2*np.sum(abs(u[tuple(s)])**2)
                s[uaxis] = 0
                if T.local_slice(True)[axis].start == 0:
                    result += np.sum(abs(u[tuple(s)])**2)
                else:
                    result += 2*np.sum(abs(u[tuple(s)])**2)
                s[uaxis] = -1
                if T.local_slice(True)[axis].stop == T.dims()[axis]:
                    result += np.sum(abs(u[tuple(s)])**2)
                else:
                    result += 2*np.sum(abs(u[tuple(s)])**2)
            else:
                s[uaxis] = slice(1, None)
                result = 2*np.sum(abs(u[tuple(s)])**2)
                s[uaxis] = 0
                if T.local_slice(True)[axis].start == 0:
                    result += np.sum(abs(u[tuple(s)])**2)
                else:
                    result += 2*np.sum(abs(u[tuple(s)])**2)

    else:
        result = np.sum(abs(u[...])**2)

    result = comm.allreduce(result)
    return result
