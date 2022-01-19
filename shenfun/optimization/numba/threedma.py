import numba as nb
from .la import Solve_axis_2D, Solve_axis_3D

__all__ = ['ThreeDMA_Solve', 'ThreeDMA_inner_solve']

def ThreeDMA_Solve(x, data, axis=0):
    n = x.ndim
    if n == 1:
        ThreeDMA_inner_solve(x, data)
    elif n == 2:
        Solve_axis_2D(data, x, ThreeDMA_inner_solve, axis)
    elif n == 3:
        Solve_axis_3D(data, x, ThreeDMA_inner_solve, axis)

@nb.jit(nopython=True, fastmath=True, cache=True)
def ThreeDMA_inner_solve(u, data):
    d = data[0, :]
    u1 = data[1, 2:]
    u2 = data[1, 4:]
    n = d.shape[0]
    u[n-1] = u[n-1]/d[n-1]
    u[n-2] = u[n-2]/d[n-2]
    u[n-3] = (u[n-3]-u1[n-3]*u[n-1])/d[n-3]
    u[n-4] = (u[n-4]-u1[n-4]*u[n-2])/d[n-4]
    for i in range(n - 5, -1, -1):
        u[i] = (u[i] - u1[i]*u[i+2] - u2[i]*u[i+4])/d[i]
