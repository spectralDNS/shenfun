import numba as nb
import numpy as np

__all__ = ['LU_Helmholtz', 'Solve_Helmholtz']

def LU_Helmholtz(A, B, A_s, B_s, neumann, d0, d1, d2, L, axis):
    n = d0.ndim
    #A_0[0] = np.pi/A_scale
    A_0, A_2, A_4, B_m2, B_0, B_2 = preLU(A, B, neumann)

    if n == 1:
        LU_Helmholtz_1D(A_0, A_2, A_4, B_m2, B_0, B_2, np.asscalar(A_s), np.asscalar(B_s), neumann, d0, d1, d2, L)
    elif n == 2:
        LU_Helmholtz_2D(A_0, A_2, A_4, B_m2, B_0, B_2, axis, A_s, B_s, neumann, d0, d1, d2, L)
    elif n == 3:
        LU_Helmholtz_3D(A_0, A_2, A_4, B_m2, B_0, B_2, axis, A_s, B_s, neumann, d0, d1, d2, L)

def Solve_Helmholtz(b, u, neumann, d0, d1, d2, L, axis):
    n = d0.ndim
    y = np.zeros(d0.shape[axis]-2).astype(u.dtype)
    if n == 1:
        Solve_Helmholtz_1D(b, u, neumann, d0, d1, d2, L, y)
    elif n == 2:
        Solve_Helmholtz_2D(b, u, neumann, d0, d1, d2, L, y, axis)
    elif n == 3:
        Solve_Helmholtz_3D(b, u, neumann, d0, d1, d2, L, y, axis)

def preLU(A, B, neumann):
    A_0 = A[0].copy()
    A_2 = A[2].copy()
    A_4 = A[4].copy()
    B_m2 = B[-2].copy()
    B_0 = B[0].copy()
    B_2 = B[2].copy()
    N = A_0.shape[0]
    if neumann:
        B_0[0] = 0.0
        for i in range(1, N):
            A_0[i] /= pow(i, 2)
            B_0[i] /= pow(i, 2)
        for i in range(2, N):
            A_2[i-2] /= pow(i, 2)
            B_2[i-2] /= pow(i, 2)
        for i in range(4, N):
            A_4[i-4] /= pow(i, 2)
        for i in range(1, N-2):
            B_m2[i] /= pow(i, 2)
    return A_0, A_2, A_4, B_m2, B_0, B_2

@nb.jit(nopython=True, fastmath=True, cache=True)
def LU_Helmholtz_1D(A_0, A_2, A_4, B_m2, B_0, B_2, A_scale, B_scale, neumann, d0, d1, d2, L):
    N = A_0.shape[0]
    if neumann:
        A_0[0] = np.pi/A_scale

    d0[0] = A_scale*A_0[0] + B_scale*B_0[0]
    d0[1] = A_scale*A_0[1] + B_scale*B_0[1]
    d1[0] = A_scale*A_2[0] + B_scale*B_2[0]
    d1[1] = A_scale*A_2[1] + B_scale*B_2[1]
    d2[0] = A_scale*A_4[0]
    d2[1] = A_scale*A_4[1]
    for i in range(2, N):
        L[i-2] = B_scale*B_m2[i-2] / d0[i-2]
        d0[i] = A_scale*A_0[i] + B_scale*B_0[i] - L[i-2]*d1[i-2]
        if i < N-2:
            d1[i] = A_scale*A_2[i] + B_scale*B_2[i] - L[i-2]*d2[i-2]
        if i < N-4:
            d2[i] = A_scale*A_4[i] - L[i-2]*d2[i-2]

@nb.jit(nopython=True, fastmath=True, cache=True)
def LU_Helmholtz_3D(A_0, A_2, A_4, B_m2, B_0, B_2,
                    axis, A_scale, B_scale,
                    neumann, d0, d1, d2, L):
    if axis == 0:
        for j in range(d0.shape[1]):
            for k in range(d0.shape[2]):
                LU_Helmholtz_1D(A_0, A_2, A_4, B_m2, B_0, B_2,
                                A_scale[0, j, k],
                                B_scale[0, j, k],
                                neumann,
                                d0[:, j, k],
                                d1[:, j, k],
                                d2[:, j, k],
                                L[:, j, k])

    elif axis == 1:
        for i in range(d0.shape[0]):
            for k in range(d0.shape[2]):
                LU_Helmholtz_1D(A_0, A_2, A_4, B_m2, B_0, B_2,
                                A_scale[i, 0, k],
                                B_scale[i, 0, k],
                                neumann,
                                d0[i, :, k],
                                d1[i, :, k],
                                d2[i, :, k],
                                L[i, :, k])

    elif axis == 2:
        for i in range(d0.shape[0]):
            for j in range(d0.shape[1]):
                LU_Helmholtz_1D(A_0, A_2, A_4, B_m2, B_0, B_2,
                                A_scale[i, j, 0],
                                B_scale[i, j, 0],
                                neumann,
                                d0[i, j, :],
                                d1[i, j, :],
                                d2[i, j, :],
                                L[i, j, :])

@nb.jit(nopython=True, fastmath=True, cache=True)
def LU_Helmholtz_2D(A_0, A_2, A_4, B_m2, B_0, B_2,
                    axis, A_scale, B_scale, neumann,
                    d0, d1, d2, L):
    if axis == 0:
        for i in range(d0.shape[1]):
            LU_Helmholtz_1D(A_0, A_2, A_4, B_m2, B_0, B_2,
                            A_scale[0, i],
                            B_scale[0, i],
                            neumann,
                            d0[:, i],
                            d1[:, i],
                            d2[:, i],
                            L[:, i])

    elif axis == 1:
        for i in range(d0.shape[0]):
            LU_Helmholtz_1D(A_0, A_2, A_4, B_m2, B_0, B_2,
                            A_scale[i, 0],
                            B_scale[i, 0],
                            neumann,
                            d0[i, :],
                            d1[i, :],
                            d2[i, :],
                            L[i, :])

@nb.jit(nopython=True, fastmath=True, cache=True)
def Solve_Helmholtz_1D(fk, u_hat, neumann, d0, d1, d2, L, y):
    sum_even = 0.0
    sum_odd = 0.0

    N = d0.shape[0]-2
    y[0] = fk[0]
    y[1] = fk[1]
    for i in range(2, N):
        y[i] = fk[i] - L[i-2]*y[i-2]

    u_hat[N-1] = y[N-1] / d0[N-1]
    u_hat[N-2] = y[N-2] / d0[N-2]
    u_hat[N-3] = (y[N-3] - d1[N-3]*u_hat[N-1]) / d0[N-3]
    u_hat[N-4] = (y[N-4] - d1[N-4]*u_hat[N-2]) / d0[N-4]
    for i in range(N-5, -1, -1):
        u_hat[i] = y[i] - d1[i]*u_hat[i+2]
        if i % 2 == 0:
            sum_even += u_hat[i+4]
            u_hat[i] -= d2[i]*sum_even
        else:
            sum_odd += u_hat[i+4]
            u_hat[i] -= d2[i]*sum_odd
        u_hat[i] /= d0[i]

    if neumann:
        u_hat[0] = 0.0
        for i in range(1, N):
            u_hat[i] /= (i*i)

@nb.jit(nopython=True, fastmath=True, cache=True)
def Solve_Helmholtz_2D(fk, u_hat, neumann, d0, d1, d2, L, y, axis):
    if axis == 0:
        for j in range(d0.shape[1]):
            Solve_Helmholtz_1D(fk[:, j], u_hat[:, j], neumann, d0[:, j],
                               d1[:, j], d2[:, j], L[:, j], y)
    elif axis == 1:
        for i in range(d0.shape[0]):
            Solve_Helmholtz_1D(fk[i], u_hat[i], neumann, d0[i], d1[i], d2[i],
                               L[i], y)

@nb.jit(nopython=True, fastmath=True, cache=True)
def Solve_Helmholtz_3D(fk, u_hat, neumann, d0, d1, d2, L, y, axis):
    if axis == 0:
        for j in range(d0.shape[1]):
            for k in range(d0.shape[2]):
                Solve_Helmholtz_1D(fk[:, j, k], u_hat[:, j, k], neumann,
                                   d0[:, j, k], d1[:, j, k], d2[:, j, k],
                                   L[:, j, k], y)
    elif axis == 1:
        for i in range(d0.shape[0]):
            for k in range(d0.shape[2]):
                Solve_Helmholtz_1D(fk[i, :, k], u_hat[i, :, k], neumann,
                                   d0[i, :, k], d1[i, :, k], d2[i, :, k],
                                   L[i, :, k], y)
    elif axis == 2:
        for i in range(d0.shape[0]):
            for j in range(d0.shape[1]):
                Solve_Helmholtz_1D(fk[i, j], u_hat[i, j], neumann,
                                   d0[i, j], d1[i, j], d2[i, j],
                                   L[i, j], y)
