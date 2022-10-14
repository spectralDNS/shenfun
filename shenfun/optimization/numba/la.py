import numpy as np
import numba as nb

__all__ = ['SolverGeneric1ND_solve_data',
           'Solve_axis_2D', 'Solve_axis_3D', 'Solve_axis_4D']

@nb.njit
def SolverGeneric1ND_solve_data(u, data, sol, naxes, is_zero_index):
    if u.ndim == 2:
        if naxes == 0:
            for i in range(u.shape[1]):
                if i == 0 and is_zero_index:
                    continue
                sol(u[:, i], data[i])

        elif naxes == 1:
            for i in range(u.shape[0]):
                if i == 0 and is_zero_index:
                    continue
                sol(u[i], data[i])

    elif u.ndim == 3:
        if naxes == 0:
            for i in range(u.shape[1]):
                for j in range(u.shape[2]):
                    if i == 0 and j == 0 and is_zero_index:
                        continue
                    sol(u[:, i, j], data[i, j])

        elif naxes == 1:
            for i in range(u.shape[0]):
                for j in range(u.shape[2]):
                    if i == 0 and j == 0 and is_zero_index:
                        continue
                    sol(u[i, :, j], data[i, j])

        elif naxes == 2:
            for i in range(u.shape[0]):
                for j in range(u.shape[1]):
                    if i == 0 and j == 0 and is_zero_index:
                        continue
                    sol(u[i, j, :], data[i, j])
    return u

@nb.jit(nopython=True, fastmath=True, cache=False)
def Solve_axis_2D(data, x, innerfun, axis):
    if axis == 0:
        for j in range(x.shape[1]):
            innerfun(x[:, j], data)
    elif axis == 1:
        for i in range(x.shape[0]):
            innerfun(x[i, :], data)

@nb.jit(nopython=True, fastmath=True, cache=False)
def Solve_axis_3D(data, x, innerfun, axis):
    if axis == 0:
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                innerfun(x[:, j, k], data)
    elif axis == 1:
        for i in range(x.shape[0]):
            for k in range(x.shape[2]):
                innerfun(x[i, :, k], data)
    elif axis == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                innerfun(x[i, j], data)

@nb.jit(nopython=True, fastmath=True, cache=False)
def Solve_axis_4D(data, x, innerfun, axis):
    if axis == 0:
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                for l in range(x.shape[3]):
                    innerfun(x[:, j, k, l], data)
    elif axis == 1:
        for i in range(x.shape[0]):
            for k in range(x.shape[2]):
                for l in range(x.shape[3]):
                    innerfun(x[i, :, k, l], data)
    elif axis == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for l in range(x.shape[3]):
                    innerfun(x[i, j, :, l], data)
    elif axis == 3:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    innerfun(x[i, j, k], data)
