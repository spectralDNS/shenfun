import numba as nb
import numpy as np

M_PI_2 = np.pi/2

__all__ = ['LU_Helmholtz', 'Solve_Helmholtz', 'Helmholtz_matvec',
           'Helmholtz_Neumann_matvec', 'Poisson_Solve_ADD',
           'Poisson_Solve_ADD_1D', 'ANN_matvec']

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

def preLU(A, B, neumann=False):
    A_0 = A[0].copy()
    A_2 = A[2].copy()
    A_4 = A[4].copy()
    B_m2 = B[-2].copy()
    B_0 = B[0].copy()
    B_2 = B[2].copy()
    N = A_0.shape[0]
    if neumann:
        k = np.arange(N)
        j2 = k**2
        j2[0] = 1
        j2 = 1/j2
        B_0 *= j2
        j2[0] = 0
        A_0 *= j2
        A_2 *= j2[2:]
        B_2 *= j2[2:]
        A_4 *= j2[4:]
        j2[0] = 1
        B_m2 *= j2[:-2]

        #B_0[0] = 0.0
        #for i in range(1, N):
        #    A_0[i] /= pow(i, 2)
        #    B_0[i] /= pow(i, 2)
        #for i in range(2, N):
        #    A_2[i-2] /= pow(i, 2)
        #    B_2[i-2] /= pow(i, 2)
        #for i in range(4, N):
        #    A_4[i-4] /= pow(i, 2)
        #for i in range(1, N-2):
        #    B_m2[i] /= pow(i, 2)

    return A_0, A_2, A_4, B_m2, B_0, B_2

@nb.jit(nopython=True, fastmath=True, cache=True)
def LU_Helmholtz_1D(A_0, A_2, A_4, B_m2, B_0, B_2, A_scale, B_scale, neumann, d0, d1, d2, L):
    N = A_0.shape[0]
    if neumann:
        if abs(B_scale) < 1e-8:
            A_0[0] = 1/A_scale

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
        if (d0[0]-1.0)*(d0[0]-1.0) < 1e-16:
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

@nb.jit(nopython=True, fastmath=True, cache=True)
def Helmholtz_matvec1D(v, b, alfa, beta, dd, ud, bd):
    # b = (alfa*A + beta*B)*v
    # For B matrix ld = ud = -pi/2
    N = dd.shape[0]
    s1 = 0.0
    s2 = 0.0

    k = N-1
    b[k] = (dd[k]*alfa + bd[k]*beta)*v[k] - M_PI_2*beta*v[k-2]
    b[k-1] = (dd[k-1]*alfa + bd[k-1]*beta)*v[k-1] - M_PI_2*beta*v[k-3]

    for k in range(N-3, 1, -1):
        p = ud[k]*alfa
        if k % 2 == 0:
            s2 += v[k+2]
            b[k] = (dd[k]*alfa + bd[k]*beta)*v[k] - M_PI_2*beta*(v[k-2] + v[k+2]) + p*s2
        else:
            s1 += v[k+2]
            b[k] = (dd[k]*alfa + bd[k]*beta)*v[k] - M_PI_2*beta*(v[k-2] + v[k+2]) + p*s1

    k = 1
    s1 += v[k+2]
    s2 += v[k+1]
    b[k] = (dd[k]*alfa + bd[k]*beta)*v[k] - M_PI_2*beta*v[k+2] + ud[k]*alfa*s1
    b[k-1] = (dd[k-1]*alfa + bd[k-1]*beta)*v[k-1] - M_PI_2*beta*v[k+1] + ud[k-1]*alfa*s2

@nb.jit(nopython=True, fastmath=True, cache=True)
def Helmholtz_matvec3D(v, b, alfa, beta, dd, ud, bd, axis):
    if axis == 0:
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                Helmholtz_matvec1D(v[:, j, k], b[:, j, k], alfa[0, j, k],
                                   beta[0, j, k], dd, ud, bd)
    elif axis == 1:
        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                Helmholtz_matvec1D(v[i, :, k], b[i, :, k], alfa[i, 0, k],
                                   beta[i, 0, k], dd, ud, bd)
    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                Helmholtz_matvec1D(v[i, j], b[i, j], alfa[i, j, 0],
                                   beta[i, j, 0], dd, ud, bd)

@nb.jit(nopython=True, fastmath=True, cache=True)
def Helmholtz_matvec2D(v, b, alfa, beta, dd, ud, bd, axis):
    if axis == 0:
        for j in range(v.shape[1]):
            Helmholtz_matvec1D(v[:, j], b[:, j], alfa[0, j],
                               beta[0, j], dd, ud, bd)
    elif axis == 1:
        for i in range(v.shape[0]):
            Helmholtz_matvec1D(v[i], b[i], alfa[i, 0],
                               beta[i, 0], dd, ud, bd)

def Helmholtz_matvec(v, b, alfa, beta, A, B, axis):
    n = v.ndim
    A_0, A_2, A_4, B_m2, B_0, B_2 = preLU(A, B)
    if n == 1:
        Helmholtz_matvec1D(v, b, np.asscalar(alfa), np.asscalar(beta), A_0, A_2, B_0)
    elif n == 2:
        Helmholtz_matvec2D(v, b, alfa, beta, A_0, A_2, B_0, axis)
    elif n == 3:
        Helmholtz_matvec3D(v, b, alfa, beta, A_0, A_2, B_0, axis)


@nb.jit(nopython=True, fastmath=True, cache=True)
def Helmholtz_Neumann_matvec1D(v, b, alfa, beta, dd, ud, bl, bd, bu):
    # b = (alfa*A + beta*B)*v
    # A matrix has diagonal dd and upper second diagonal at ud
    # B matrix has diagonal bd and second upper and lower diagonals bu and bl
    N = dd.shape[0]
    s1 = 0.0
    s2 = 0.0
    for k in (N-1, N-2):
        b[k] = (dd[k]*alfa + bd[k]*beta)*v[k]*k**2 + bl[k-2]*beta*v[k-2]*(k-2)**2

    for k in range(N-3, 1, -1):
        p = ud[k]*alfa
        if k % 2 == 0:
            s2 += v[k+2]*(k+2)**2
            b[k] = (dd[k]*alfa + bd[k]*beta)*v[k]*k**2 + beta*(bl[k-2]*v[k-2]*(k-2)**2 + bu[k]*v[k+2]*(k+2)**2) + p*s2
        else:
            s1 += v[k+2]*(k+2)**2
            b[k] = (dd[k]*alfa + bd[k]*beta)*v[k]*k**2 + beta*(bl[k-2]*v[k-2]*(k-2)**2 + bu[k]*v[k+2]*(k+2)**2) + p*s1

    k = 1
    s1 += v[k+2]*(k+2)**2
    b[k] = (dd[k]*alfa + bd[k]*beta)*v[k]*k**2 + beta*(bu[k]*v[k+2]*(k+2)**2) + ud[k]*alfa*s1
    k = 0
    s2 += v[k+2]*(k+2)**2
    b[k] = (dd[k]*alfa + bd[k]*beta)*v[k]*k**2 + beta*(bu[k]*v[k+2]*(k+2)**2) + ud[k]*alfa*s2
    b[0] += bd[0]*v[0]*beta
    b[2] += bl[0]*v[0]*beta

@nb.jit(nopython=True, fastmath=True, cache=True)
def Helmholtz_Neumann_matvec1D2(v, b, alfa, beta, dd, ud, bl, bd, bu):
    # b = (alfa*A + beta*B)*v
    # A matrix has diagonal dd and upper second diagonal at ud
    # B matrix has diagonal bd and second upper and lower diagonals bu and bl
    # Alternative implementation
    N = dd.shape[0]
    s1 = 0.0
    s2 = 0.0
    for k in (N-1, N-2):
        b[k] = (-dd[k] + bd[k]*beta)*v[k] + bl[k-2]*beta*v[k-2]

    j = 0
    for k in range(N-3, 1, -1):
        j += 1
        p = ud[k]/(k+2)**2
        if k % 2 == 0:
            s2 += v[k+2]*(k+2)**2
            b[k] = (-dd[k] + bd[k]*beta)*v[k] + beta*(bl[k-2]*v[k-2]) + bu[k]*v[k+2] - p*s2
        else:
            s1 += v[k+2]*(k+2)**2
            b[k] = (-dd[k] + bd[k]*beta)*v[k] + beta*(bl[k-2]*v[k-2]) + bu[k]*v[k+2] - p*s1

    k = 1
    s1 += v[k+2]*(k+2)**2
    b[k] = (-dd[k] + bd[k]*beta)*v[k] + beta*(bu[k]*v[k+2]) - ud[k]/(k+2)**2*s1
    k = 0
    s2 += v[k+2]*(k+2)**2
    b[k] = (-dd[k] + bd[k]*beta)*v[k] + beta*(bu[k]*v[k+2]) - ud[k]/(k+2)**2*s2
    b[0] += bd[0]*v[0]*beta
    b[2] += bl[0]*v[0]*beta

@nb.jit(nopython=True, fastmath=True, cache=True)
def Helmholtz_Neumann_matvec3D(v, b, alfa, beta, dd, ud, bl, bd, bu, axis):
    if axis == 0:
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                Helmholtz_Neumann_matvec1D(v[:, j, k], b[:, j, k], alfa[0, j, k],
                                           beta[0, j, k], dd, ud, bl, bd, bu)
    elif axis == 1:
        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                Helmholtz_Neumann_matvec1D(v[i, :, k], b[i, :, k], alfa[i, 0, k],
                                           beta[i, 0, k], dd, ud, bl, bd, bu)
    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                Helmholtz_Neumann_matvec1D(v[i, j], b[i, j], alfa[i, j, 0],
                                           beta[i, j, 0], dd, ud, bl, bd, bu)

@nb.jit(nopython=True, fastmath=True, cache=True)
def Helmholtz_Neumann_matvec2D(v, b, alfa, beta, dd, ud, bl, bd, bu, axis):
    if axis == 0:
        for j in range(v.shape[1]):
            Helmholtz_Neumann_matvec1D(v[:, j], b[:, j], alfa[0, j],
                                       beta[0, j], dd, ud, bl, bd, bu)
    elif axis == 1:
        for i in range(v.shape[0]):
            Helmholtz_Neumann_matvec1D(v[i], b[i], alfa[i, 0],
                                       beta[i, 0], dd, ud, bl, bd, bu)

def Helmholtz_Neumann_matvec(v, b, alfa, beta, A, B, axis):
    n = v.ndim
    A_0, A_2, A_4, B_m2, B_0, B_2 = preLU(A, B, True)
    if n == 1:
        Helmholtz_Neumann_matvec1D(v.v, b.v, alfa.item(), beta.item(),
                                   A_0, A_2, B_m2, B_0, B_2)
    elif n == 2:
        Helmholtz_Neumann_matvec2D(v.v, b.v, alfa, beta,
                                   A_0, A_2, B_m2, B_0, B_2, axis)
    elif n == 3:
        Helmholtz_Neumann_matvec3D(v.v, b.v, alfa, beta,
                                   A_0, A_2, B_m2, B_0, B_2, axis)

def Poisson_Solve_ADD(A, b, u, axis=0):
    n = u.ndim
    if n == 1:
        u = Poisson_Solve_ADD_1D(A[0], A[2], A.scale, b, u)
    elif n == 2:
        u = Poisson_Solve_ADD_2D(A[0], A[2], A.scale, b, u, axis)
    elif n == 3:
        u = Poisson_Solve_ADD_3D(A[0], A[2], A.scale, b, u, axis)
    return u

@nb.jit(nopython=True, fastmath=True, cache=True)
def Poisson_Solve_ADD_1D(d, d1, scale, b, u):
    N = d.shape[0]
    se = u[0]*0
    so = u[0]*0
    u[N-1] = b[N-1] / d[N-1]
    u[N-2] = b[N-2] / d[N-2]
    for k in range(N-3, -1, -1):
        if k%2 == 0:
            se += u[k+2]
            u[k] = b[k] - d1[k]*se
        else:
            so += u[k+2]
            u[k] = b[k] - d1[k]*so
        u[k] /= d[k]

    if not abs(scale-1) < 1e-8:
        for k in range(N):
            u[k] /= scale
    return u

@nb.jit(nopython=True, fastmath=True, cache=True)
def Poisson_Solve_ADD_2D(d, d1, scale, b, u, axis):
    if axis == 0:
        for j in range(u.shape[1]):
            u[:, j] = Poisson_Solve_ADD_1D(d, d1, scale, b[:, j], u[:, j])
    elif axis == 1:
        for i in range(u.shape[0]):
            u[i] = Poisson_Solve_ADD_1D(d, d1, scale, b[i], u[i])
    return u

@nb.jit(nopython=True, fastmath=True, cache=True)
def Poisson_Solve_ADD_3D(d, d1, scale, b, u, axis):
    if axis == 0:
        for j in range(u.shape[1]):
            for k in range(u.shape[2]):
                u[:, j, k] = Poisson_Solve_ADD_1D(d, d1, scale, b[:, j, k], u[:, j, k])

    elif axis == 1:
        for i in range(u.shape[0]):
            for k in range(u.shape[2]):
                u[i, :, k] = Poisson_Solve_ADD_1D(d, d1, scale, b[i, :, k], u[i, :, k])

    elif axis == 2:
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                u[i, j] = Poisson_Solve_ADD_1D(d, d1, scale, b[i, j], u[i, j])
    return u

@nb.jit(nopython=True, fastmath=True, cache=True)
def ANN_matvec1D(v, c, d0, d2, scale=1):
    N = v.shape[0]-2
    c[N-1] = d0[N-1]*v[N-1]
    c[N-2] = d0[N-2]*v[N-2]
    s0 = 0
    s1 = 0
    for k in range(N-3, -1, -1):
        c[k] = d0[k]*v[k]
        if (k % 2) == 0:
            s0 += v[k+2]*(k+2)**2
            c[k] += s0*d2[k]/(k+2)**2
        else:
            s1 += v[k+2]*(k+2)**2
            c[k] += s1*d2[k]/(k+2)**2
    if scale != 1:
        c /= scale

@nb.jit(nopython=True, fastmath=True, cache=True)
def ANN_matvec2D(v, b, d0, d2, scale, axis):
    if axis == 0:
        for j in range(v.shape[1]):
            ANN_matvec1D(v[:, j], b[:, j], d0, d2, scale)
    elif axis == 1:
        for i in range(v.shape[0]):
            ANN_matvec1D(v[i], b[i], d0, d2, scale)

@nb.jit(nopython=True, fastmath=True, cache=True)
def ANN_matvec3D(v, b, d0, d2, scale, axis):
    if axis == 0:
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                ANN_matvec1D(v[:, j, k], b[:, j, k], d0, d2, scale)
    elif axis == 1:
        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                ANN_matvec1D(v[i, :, k], b[i, :, k], d0, d2, scale)
    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                ANN_matvec1D(v[i, j], b[i, j], d0, d2, scale)

def ANN_matvec(v, b, A, axis=0):
    n = v.ndim
    vc = v.__array__()
    bc = b.__array__()
    if n == 1:
        ANN_matvec1D(vc, bc, A[0], A[2], A.scale)
    elif n == 2:
        ANN_matvec2D(vc, bc, A[0], A[2], A.scale, axis)
    elif n == 3:
        ANN_matvec3D(vc, bc, A[0], A[2], A.scale, axis)
    return b
