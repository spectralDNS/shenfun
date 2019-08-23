import numpy as np
import numba as nb
from .tdma import *
from .pdma import *
from .helmholtz import *
from .biharmonic import *

@nb.jit(nopython=True, fastmath=True, cache=True)
def outer2D(a, b, c, symmetric):
    N, M = a.shape[1:]
    if symmetric:
        for i in range(N):
            for j in range(M):
                c[0, i, j] = a[0, i, j]**2           # (0, 0)
                c[1, i, j] = a[0, i, j]*a[1, i, j]   # (0, 1)
                c[2, i, j] = c[1, i, j]              # (1, 0)
                c[3, i, j] = a[1, i, j]**2           # (1, 1)
    else:
        for i in range(N):
            for j in range(M):
                c[0, i, j] = a[0, i, j]*b[0, i, j]   # (0, 0)
                c[1, i, j] = a[0, i, j]*b[1, i, j]   # (0, 1)
                c[2, i, j] = a[1, i, j]*b[0, i, j]   # (1, 0)
                c[3, i, j] = a[1, i, j]*b[1, i, j]   # (1, 1)


@nb.jit(nopython=True, fastmath=True, cache=True)
def outer3D(a, b, c, symmetric):
    N, M = a.shape[1:]
    if symmetric:
        for i in range(N):
            for j in range(M):
                c[0, i, j] = a[0, i, j]**2           # (0, 0)
                c[1, i, j] = a[0, i, j]*a[1, i, j]   # (0, 1)
                c[2, i, j] = a[0, i, j]*a[2, i, j]   # (0, 2)
                c[3, i, j] = c[1, i, j]              # (1, 0)
                c[4, i, j] = a[1, i, j]**2           # (1, 1)
                c[5, i, j] = a[1, i, j]*a[2, i, j]   # (1, 2)
                c[6, i, j] = c[2, i, j]              # (2, 0)
                c[7, i, j] = c[5, i, j]              # (2, 1)
                c[8, i, j] = a[2, i, j]**2           # (2, 2)
    else:
        for i in range(N):
            for j in range(M):
                c[0, i, j] = a[0, i, j]*b[0, i, j]   # (0, 0)
                c[1, i, j] = a[0, i, j]*b[1, i, j]   # (0, 1)
                c[2, i, j] = a[0, i, j]*b[2, i, j]   # (0, 2)
                c[3, i, j] = a[1, i, j]*b[0, i, j]   # (1, 0)
                c[4, i, j] = a[1, i, j]*b[1, i, j]   # (1, 1)
                c[5, i, j] = a[1, i, j]*b[2, i, j]   # (1, 2)
                c[6, i, j] = a[2, i, j]*b[0, i, j]   # (2, 0)
                c[7, i, j] = a[2, i, j]*b[1, i, j]   # (2, 1)
                c[8, i, j] = a[2, i, j]*b[2, i, j]   # (2, 2)


def apply_mask(u_hat, mask):
    if mask is not None:
        if u_hat.ndim == mask.ndim:
            if mask.ndim == 1:
                u_hat = apply_mask_1D(u_hat, mask)
            elif mask.ndim == 2:
                u_hat = apply_mask_2D(u_hat, mask)
            elif mask.ndim == 3:
                u_hat = apply_mask_3D(u_hat, mask)
            elif mask.ndim == 4:
                u_hat = apply_mask_4D(u_hat, mask)
            else:
                u_hat *= mask
        elif u_hat.ndim == mask.ndim + 1:
            if mask.ndim == 1:
                u_hat = apply_bmask_1D(u_hat, mask)
            elif mask.ndim == 2:
                u_hat = apply_bmask_2D(u_hat, mask)
            elif mask.ndim == 3:
                u_hat = apply_bmask_3D(u_hat, mask)
            elif mask.ndim == 4:
                u_hat = apply_bmask_4D(u_hat, mask)
            else:
                u_hat *= mask
        else:
            u_hat = apply_bxmask(u_hat, mask)
    return u_hat

@nb.jit(nopython=True, fastmath=True, cache=True)
def apply_mask_1D(u, mask):
    for i in range(mask.shape[0]):
        if mask[i] == 0:
            u[i] = 0
    return u

@nb.jit(nopython=True, fastmath=True, cache=True)
def apply_mask_2D(u, mask):
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 0:
                u[i, j] = 0
    return u

@nb.jit(nopython=True, fastmath=True, cache=True)
def apply_mask_3D(u, mask):
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i, j, k] == 0:
                    u[i, j, k] = 0
    return u

@nb.jit(nopython=True, fastmath=True, cache=True)
def apply_mask_4D(u, mask):
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                for l in range(mask.shape[3]):
                    if mask[i, j, k, l] == 0:
                        u[i, j, k, l] = 0
    return u

@nb.jit(nopython=True, fastmath=True, cache=True)
def apply_bmask_1D(u, mask):
    for j in range(u.shape[1]):
        if mask[j] == 0:
            for i in range(u.shape[0]):
                u[i, j] = 0
    return u

@nb.jit(nopython=True, fastmath=True, cache=True)
def apply_bmask_2D(u, mask):
    for j in range(mask.shape[0]):
        for k in range(mask.shape[1]):
            if mask[j, k] == 0:
                for i in range(u.shape[0]):
                    u[i, j, k] = 0
    return u

@nb.jit(nopython=True, fastmath=True, cache=True)
def apply_bmask_3D(u, mask):
    for j in range(mask.shape[0]):
        for k in range(mask.shape[1]):
            for l in range(mask.shape[2]):
                if mask[j, k, l] == 0:
                    for i in range(u.shape[0]):
                        u[i, j, k, l] = 0
    return u

@nb.jit(nopython=True, fastmath=True, cache=True)
def apply_bmask_4D(u, mask):
    for j in range(mask.shape[0]):
        for k in range(mask.shape[1]):
            for l in range(mask.shape[2]):
                for m in range(mask.shape[3]):
                    if mask[j, k, l, m] == 0:
                        for i in range(u.shape[0]):
                            u[i, j, k, l, m] = 0
    return u


@nb.jit(nopython=False, fastmath=True, cache=True)
def apply_bxmask(u_hat, mask):
    if mask is not None:
        N = mask.shape
        if len(N) == 1:
            for i in range(N[0]):
                if mask[i] == 0:
                    u_hat[..., i] = 0
        elif len(N) == 2:
            for i in range(N[0]):
                for j in range(N[1]):
                    if mask[i, j] == 0:
                        u_hat[..., i, j] = 0
        elif len(N) == 3:
            for i in range(N[0]):
                for j in range(N[1]):
                    for k in range(N[2]):
                        if mask[i, j, k] == 0:
                            u_hat[..., i, j, k] = 0
        elif len(N) == 4:
            for i in range(N[0]):
                for j in range(N[1]):
                    for k in range(N[2]):
                        for l in range(N[3]):
                            if mask[i, j, k, l] == 0:
                                u_hat[..., i, j, k, l] = 0
        else:
            u_hat *= mask
    return u_hat
