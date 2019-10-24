#cython: boundscheck=False
#cython: wraparound=False
#cython: language_level=3

import numpy as np
cimport cython
cimport numpy as np

ctypedef fused T:
    np.float64_t
    np.complex128_t

ctypedef np.int64_t int_t

def apply_mask(u_hat, mask):
    if mask is not None:
        if u_hat.ndim == mask.ndim:
            mask = np.broadcast_to(mask, u_hat.shape).copy()
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
            mask = np.broadcast_to(mask, u_hat.shape[1:]).copy()
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
        elif u_hat.ndim == mask.ndim + 2:
            mask = np.broadcast_to(mask, u_hat.shape[2:]).copy()
            if mask.ndim == 1:
                u_hat = apply_b2mask_1D(u_hat, mask)
            elif mask.ndim == 2:
                u_hat = apply_b2mask_2D(u_hat, mask)
            elif mask.ndim == 3:
                u_hat = apply_b2mask_3D(u_hat, mask)
            elif mask.ndim == 4:
                u_hat = apply_b2mask_4D(u_hat, mask)
            else:
                u_hat *= mask
        else:
            u_hat *= mask
    return u_hat

def apply_mask_1D(T[::1] u, int_t[::1] mask):
    cdef int i
    for i in range(mask.shape[0]):
        if mask[i] == 0:
            u[i] = 0
    return u

def apply_mask_2D(T[:, ::1] u, int_t[:, ::1] mask):
    cdef int i, j
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 0:
                u[i, j] = 0
    return u

def apply_mask_3D(T[:, :, ::1] u, int_t[:, :, ::1] mask):
    cdef int i, j, k
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i, j, k] == 0:
                    u[i, j, k] = 0
    return u

def apply_mask_4D(T[:, :, :, ::1] u, int_t[:, :, :, ::1] mask):
    cdef int i, j, k, l
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                for l in range(mask.shape[3]):
                    if mask[i, j, k, l] == 0:
                        u[i, j, k, l] = 0
    return u

def apply_bmask_1D(T[:, ::1] u, int_t[::1] mask):
    cdef int i, j
    for j in range(u.shape[1]):
        if mask[j] == 0:
            for i in range(u.shape[0]):
                u[i, j] = 0
    return u

def apply_bmask_2D(T[:, :, ::1] u, int_t[:, ::1] mask):
    cdef int i, j, k
    for j in range(mask.shape[0]):
        for k in range(mask.shape[1]):
            if mask[j, k] == 0:
                for i in range(u.shape[0]):
                    u[i, j, k] = 0
    return u

def apply_bmask_3D(T[:, :, :, ::1] u, int_t[:, :, ::1] mask):
    cdef int i, j, k, l
    for j in range(mask.shape[0]):
        for k in range(mask.shape[1]):
            for l in range(mask.shape[2]):
                if mask[j, k, l] == 0:
                    for i in range(u.shape[0]):
                        u[i, j, k, l] = 0
    return u

def apply_bmask_4D(T[:, :, :, :, ::1] u, int_t[:, :, :, ::1] mask):
    cdef int i, j, k, l, m
    for j in range(mask.shape[0]):
        for k in range(mask.shape[1]):
            for l in range(mask.shape[2]):
                for m in range(mask.shape[3]):
                    if mask[j, k, l, m] == 0:
                        for i in range(u.shape[0]):
                            u[i, j, k, l, m] = 0
    return u

def apply_b2mask_1D(T[:, :, ::1] u, int_t[::1] mask):
    cdef int i, j, k
    for k in range(u.shape[1]):
        if mask[k] == 0:
            for i in range(u.shape[0]):
                for j in range(u.shape[1]):
                    u[i, j, k] = 0
    return u

def apply_b2mask_2D(T[:, :, :, ::1] u, int_t[:, ::1] mask):
    cdef int i, j, k, l
    for k in range(mask.shape[0]):
        for l in range(mask.shape[1]):
            if mask[k, l] == 0:
                for i in range(u.shape[0]):
                    for j in range(u.shape[1]):
                        u[i, j, k, l] = 0
    return u

def apply_b2mask_3D(T[:, :, :, :, ::1] u, int_t[:, :, ::1] mask):
    cdef int i, j, k, l, m
    for k in range(mask.shape[0]):
        for l in range(mask.shape[1]):
            for m in range(mask.shape[2]):
                if mask[k, l, m] == 0:
                    for i in range(u.shape[0]):
                        for j in range(u.shape[2]):
                            u[i, j, k, l, m] = 0
    return u

def apply_b2mask_4D(T[:, :, :, :, :, ::1] u, int_t[:, :, :, ::1] mask):
    cdef int i, j, k, l, m, n
    for k in range(mask.shape[0]):
        for l in range(mask.shape[1]):
            for m in range(mask.shape[2]):
                for n in range(mask.shape[3]):
                    if mask[k, l, m, n] == 0:
                        for i in range(u.shape[0]):
                            for j in range(u.shape[1]):
                                u[i, j, k, l, m, n] = 0
    return u
