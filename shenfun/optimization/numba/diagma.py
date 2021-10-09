import numba as nb
from .la import Solve_axis_2D, Solve_axis_3D

__all__ = ['DiagMA_inner_solve', 'DiagMA_Solve']

def DiagMA_Solve(x, data, axis=0):
    n = x.ndim
    if n == 1:
        DiagMA_inner_solve(x, data)
    elif n == 2:
        Solve_axis_2D(data, x, DiagMA_inner_solve, axis)
    elif n == 3:
        Solve_axis_3D(data, x, DiagMA_inner_solve, axis)

@nb.njit
def DiagMA_inner_solve(u, data):
    d = data[0]
    for i in range(d.shape[0]):
        u[i] /= d[i]
