import numba as nb
import numpy as np

__all__ = ['LU_Biharmonic', 'Biharmonic_factor_pr', 'Biharmonic_Solve',
           'Biharmonic_matvec']

def LU_Biharmonic(a0, alfa, beta, sii, siu, siuu, ail, aii, aiu,
                  bill, bil, bii, biu, biuu, u0, u1,
                  u2, l0, l1, axis):
    if l1.ndim == 2:
        LU_Biharmonic_1D(a0, np.atleast_1d(alfa).item(), np.atleast_1d(beta).item(),
                         sii, siu, siuu, ail, aii, aiu,
                         bill, bil, bii, biu, biuu, u0, u1,
                         u2, l0, l1)
    elif l1.ndim == 3:
        LU_Biharmonic_2D(a0, alfa, beta, sii, siu, siuu, ail, aii, aiu,
                         bill, bil, bii, biu, biuu, u0, u1,
                         u2, l0, l1, axis)
    elif l1.ndim == 4:
        LU_Biharmonic_3D(a0, alfa, beta, sii, siu, siuu, ail, aii, aiu,
                         bill, bil, bii, biu, biuu, u0, u1,
                         u2, l0, l1, axis)

@nb.jit(nopython=True, fastmath=True, cache=True)
def LU_Biharmonic_2D(a0, alfa, beta, sii, siu, siuu, ail, aii, aiu,
                     bill, bil, bii, biu, biuu, u0, u1,
                     u2, l0, l1, axis):
    if axis == 0:
        for j in range(l1.shape[2]):
            LU_Biharmonic_1D(a0, alfa[0, j], beta[0, j],
                             sii, siu, siuu, ail, aii, aiu,
                             bill, bil, bii, biu, biuu,
                             u0[:, :, j], u1[:, :, j],
                             u2[:, :, j], l0[:, :, j], l1[:, :, j])
    elif axis == 1:
        for i in range(l1.shape[1]):
            LU_Biharmonic_1D(a0, alfa[i, 0], beta[i, 0],
                             sii, siu, siuu, ail, aii, aiu,
                             bill, bil, bii, biu, biuu,
                             u0[:, i], u1[:, i],
                             u2[:, i], l0[:, i], l1[:, i])

@nb.jit(nopython=True, fastmath=True, cache=True)
def LU_Biharmonic_3D(a0, alfa, beta, sii, siu, siuu, ail, aii, aiu,
                     bill, bil, bii, biu, biuu, u0, u1,
                     u2, l0, l1, axis):
    if axis == 0:
        for j in range(l1.shape[2]):
            for k in range(l1.shape[3]):
                LU_Biharmonic_1D(a0, alfa[0, j, k], beta[0, j, k],
                                 sii, siu, siuu, ail, aii, aiu,
                                 bill, bil, bii, biu, biuu,
                                 u0[:, :, j, k], u1[:, :, j, k],
                                 u2[:, :, j, k], l0[:, :, j, k], l1[:, :, j, k])
    elif axis == 1:
        for i in range(l1.shape[1]):
            for k in range(l1.shape[3]):
                LU_Biharmonic_1D(a0, alfa[i, 0, k], beta[i, 0, k],
                                 sii, siu, siuu, ail, aii, aiu,
                                 bill, bil, bii, biu, biuu,
                                 u0[:, i, :, k], u1[:, i, :, k],
                                 u2[:, i, :, k], l0[:, i, :, k], l1[:, i, :, k])
    elif axis == 2:
        for i in range(l1.shape[1]):
            for j in range(l1.shape[2]):
                LU_Biharmonic_1D(a0, alfa[i, j, 0], beta[i, j, 0],
                                 sii, siu, siuu, ail, aii, aiu,
                                 bill, bil, bii, biu, biuu,
                                 u0[:, i, j], u1[:, i, j],
                                 u2[:, i, j], l0[:, i, j], l1[:, i, j])

@nb.jit(nopython=True, fastmath=True, cache=True)
def LU_Biharmonic_1D(a, b, c,
                     # 3 upper diagonals of SBB
                     sii, siu, siuu,
                     # All 3 diagonals of ABB
                     ail, aii, aiu,
                     # All 5 diagonals of BBB
                     bill, bil, bii, biu, biuu,
                     # Three upper and two lower diagonals of LU decomposition
                     u0, u1, u2, l0, l1):
    LU_oe_Biharmonic_1D(0, a, b, c, sii[::2], siu[::2], siuu[::2], ail[::2], aii[::2], aiu[::2], bill[::2], bil[::2], bii[::2], biu[::2], biuu[::2], u0[0], u1[0], u2[0], l0[0], l1[0])
    LU_oe_Biharmonic_1D(1, a, b, c, sii[1::2], siu[1::2], siuu[1::2], ail[1::2], aii[1::2], aiu[1::2], bill[1::2], bil[1::2], bii[1::2], biu[1::2], biuu[1::2], u0[1], u1[1], u2[1], l0[1], l1[1])

@nb.jit(nopython=True, fastmath=True, cache=True)
def LU_oe_Biharmonic_1D(odd, a, b, c,
                        # 3 upper diagonals of SBB
                        sii, siu, siuu,
                        # All 3 diagonals of ABB
                        ail, aii, aiu,
                        # All 5 diagonals of BBB
                        bill, bil, bii, biu, biuu,
                        # Two upper and two lower diagonals of LU decomposition
                        u0, u1, u2, l0, l1):

    M = sii.shape[0]
    c0 = np.zeros(M)
    c1 = np.zeros(M)
    c2 = np.zeros(M)

    c0[0] = a*sii[0] + b*aii[0] + c*bii[0]
    c0[1] = a*siu[0] + b*aiu[0] + c*biu[0]
    c0[2] = a*siuu[0] + c*biuu[0]
    m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(6+odd+2)*(6+odd+2))
    c0[3] = m*a*np.pi/(6+odd+3.)
    #c0[3] = a*8./(6+odd+3.)*pi*(odd+1.)*(odd+2.)*(odd*(odd+4.)+3.*pow(6+odd+2., 2))
    m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(8+odd+2)*(8+odd+2))
    c0[4] = m*a*np.pi/(8+odd+3.)
    #c0[4] = a*8./(8+odd+3.)*pi*(odd+1.)*(odd+2.)*(odd*(odd+4.)+3.*pow(8+odd+2., 2))
    c1[0] = b*ail[0] + c*bil[0]
    c1[1] = a*sii[1] + b*aii[1] + c*bii[1]
    c1[2] = a*siu[1] + b*aiu[1] + c*biu[1]
    c1[3] = a*siuu[1] + c*biuu[1]
    m = 8*(odd+3)*(odd+4)*((odd+2)*(odd+6)+3*(8+odd+2)*(8+odd+2))
    c1[4] = m*a*np.pi/(8+odd+3.)
    #c1[4] = a*8./(8+odd+3.)*pi*(odd+3.)*(odd+4.)*((odd+2.)*(odd+6.)+3.*pow(8+odd+2., 2))
    c2[0] = c*bill[0]
    c2[1] = b*ail[1] + c*bil[1]
    c2[2] = a*sii[2] + b*aii[2] + c*bii[2]
    c2[3] = a*siu[2] + b*aiu[2] + c*biu[2]
    c2[4] = a*siuu[2] + c*biuu[2]
    for i in range(5, M):
        j = 2*i+odd
        m = 8*(odd+1)*(odd+2)*(odd*(odd+4)+3*(j+2)*(j+2))
        c0[i] = m*a*np.pi/(j+3.)
        m = 8*(odd+3)*(odd+4)*((odd+2)*(odd+6)+3*(j+2)*(j+2))
        c1[i] = m*a*np.pi/(j+3.)
        m = 8*(odd+5)*(odd+6)*((odd+4)*(odd+8)+3*(j+2)*(j+2))
        c2[i] = m*a*np.pi/(j+3.)
        #c0[i] = a*8./(j+3.)*pi*(odd+1.)*(odd+2.)*(odd*(odd+4.)+3.*pow(j+2., 2))
        #c1[i] = a*8./(j+3.)*pi*(odd+3.)*(odd+4.)*((odd+2)*(odd+6.)+3.*pow(j+2., 2))
        #c2[i] = a*8./(j+3.)*pi*(odd+5.)*(odd+6.)*((odd+4)*(odd+8.)+3.*pow(j+2., 2))

    u0[0] = c0[0]
    u1[0] = c0[1]
    u2[0] = c0[2]
    for kk in range(1, M):
        l0[kk-1] = c1[kk-1]/u0[kk-1]
        if kk < M-1:
            l1[kk-1] = c2[kk-1]/u0[kk-1]

        for i in range(kk, M):
            c1[i] = c1[i] - l0[kk-1]*c0[i]

        if kk < M-1:
            for i in range(kk, M):
                c2[i] = c2[i] - l1[kk-1]*c0[i]

        for i in range(kk, M):
            c0[i] = c1[i]
            c1[i] = c2[i]

        if kk < M-2:
            c2[kk] = c*bill[kk]
            c2[kk+1] = b*ail[kk+1] + c*bil[kk+1]
            c2[kk+2] = a*sii[kk+2] + b*aii[kk+2] + c*bii[kk+2]
            if kk < M-3:
                c2[kk+3] = a*siu[kk+2] + b*aiu[kk+2] + c*biu[kk+2]
            if kk < M-4:
                c2[kk+4] = a*siuu[kk+2] + c*biuu[kk+2]
            if kk < M-5:
                k = 2*(kk+2)+odd
                for i in range(kk+5, M):
                    j = 2*i+odd
                    m = 8*(k+1)*(k+2)*(k*(k+4)+3*(j+2)*(j+2))
                    c2[i] = m*a*np.pi/(j+3.)
                    #c2[i] = a*8./(j+3.)*pi*(k+1.)*(k+2.)*(k*(k+4.)+3.*pow(j+2., 2))

        u0[kk] = c0[kk]
        if kk < M-1:
            u1[kk] = c0[kk+1]
        if kk < M-2:
            u2[kk] = c0[kk+2]

@nb.jit(nopython=True, fastmath=True, cache=True)
def Biharmonic_factor_pr_3D(axis, a, b, l0, l1):
    if axis == 0:
        for ii in range(a.shape[2]):
            for jj in range(a.shape[3]):
                Biharmonic_factor_pr_1D(a[:, :, ii, jj],
                                        b[:, :, ii, jj],
                                        l0[:, :, ii, jj],
                                        l1[:, :, ii, jj])
    elif axis == 1:
        for ii in range(a.shape[1]):
            for jj in range(a.shape[3]):
                Biharmonic_factor_pr_1D(a[:, ii, :, jj],
                                        b[:, ii, :, jj],
                                        l0[:, ii, :, jj],
                                        l1[:, ii, :, jj])

    elif axis == 2:
        for ii in range(a.shape[1]):
            for jj in range(a.shape[2]):
                Biharmonic_factor_pr_1D(a[:, ii, jj, :],
                                        b[:, ii, jj, :],
                                        l0[:, ii, jj, :],
                                        l1[:, ii, jj, :])

@nb.jit(nopython=True, fastmath=True, cache=True)
def Biharmonic_factor_pr_2D(axis, a, b, l0, l1):
    if axis == 0:
        for ii in range(a.shape[2]):
            Biharmonic_factor_pr_1D(a[:, :, ii],
                                    b[:, :, ii],
                                    l0[:, :, ii],
                                    l1[:, :, ii])
    elif axis == 1:
        for ii in range(a.shape[1]):
            Biharmonic_factor_pr_1D(a[:, ii, :],
                                    b[:, ii, :],
                                    l0[:, ii, :],
                                    l1[:, ii, :])

@nb.jit(nopython=True, fastmath=True, cache=True)
def Biharmonic_factor_pr_1D(a, b, l0, l1):
    Biharmonic_factor_oe_pr(0, a[0], b[0], l0[0], l1[0])
    Biharmonic_factor_oe_pr(1, a[1], b[1], l0[1], l1[1])

@nb.jit(nopython=True, fastmath=True, cache=True)
def Biharmonic_factor_oe_pr(odd, a, b, l0, l1):
    M = l0.shape[0]+1
    k = odd
    a[0] = 8*k*(k+1)*(k+2)*(k+4)*np.pi
    b[0] = 24*(k+1)*(k+2)*np.pi
    k = 2+odd
    a[1] = 8*k*(k+1)*(k+2)*(k+4)*np.pi - l0[0]*a[0]
    b[1] = 24*(k+1)*(k+2)*np.pi - l0[0]*b[0]
    for k in range(2, M-3):
        kk = 2*k+odd
        pp = 8*kk*(kk+1)*(kk+2)*(kk+4)
        rr = 24*(kk+1)*(kk+2)
        a[k] = pp*np.pi - l0[k-1]*a[k-1] - l1[k-2]*a[k-2]
        b[k] = rr*np.pi - l0[k-1]*b[k-1] - l1[k-2]*b[k-2]

def Biharmonic_factor_pr(a, b, l0, l1, axis):
    if a.ndim == 2:
        Biharmonic_factor_pr_1D(a, b, l0, l1)
    elif a.ndim == 3:
        Biharmonic_factor_pr_2D(axis, a, b, l0, l1)
    elif a.ndim == 4:
        Biharmonic_factor_pr_3D(axis, a, b, l0, l1)

def Biharmonic_Solve(b, u, u0, u1, u2, l0, l1, ak, bk, a0, axis=0):
    if b.ndim == 1:
        Solve_Biharmonic_1D(b, u, u0, u1, u2, l0, l1, ak, bk, a0)
    elif b.ndim == 2:
        Solve_Biharmonic_2D(axis, b, u, u0, u1, u2, l0, l1, ak, bk, a0)
    elif b.ndim == 3:
        Solve_Biharmonic_3D(axis, b, u, u0, u1, u2, l0, l1, ak, bk, a0)

@nb.jit(nopython=True, fastmath=True, cache=True)
def Solve_Biharmonic_1D(fk, uk, u0, u1, u2, l0, l1, a, b, ac):
    Solve_oe_Biharmonic_1D(0, fk[::2], uk[::2], u0[0], u1[0], u2[0], l0[0], l1[0], a[0], b[0], ac)
    Solve_oe_Biharmonic_1D(1, fk[1::2], uk[1::2], u0[1], u1[1], u2[1], l0[1], l1[1], a[1], b[1], ac)

@nb.jit(nopython=True, fastmath=True, cache=True)
def BackBsolve_U(M, odd, f, uk, u0, u1, u2, l0, l1, a, b, ac):
    s1 = 0.0
    s2 = 0.0

    uk[M-1] = f[M-1] / u0[M-1]
    uk[M-2] = (f[M-2] - u1[M-2]*uk[M-1]) / u0[M-2]
    uk[M-3] = (f[M-3] - u1[M-3]*uk[M-2] - u2[M-3]*uk[M-1]) / u0[M-3]

    for kk in range(M-4, -1, -1):
        k = 2*kk+odd
        j = k+6
        s1 += uk[kk+3]/(j+3.)
        s2 += (uk[kk+3]/(j+3.))*((j+2)*(j+2))
        uk[kk] = (f[kk] - u1[kk]*uk[kk+1] - u2[kk]*uk[kk+2] - a[kk]*ac*s1 - b[kk]*ac*s2) / u0[kk]

@nb.jit(nopython=True, fastmath=True, cache=True)
def Solve_oe_Biharmonic_1D(odd, fk, uk, u0, u1, u2, l0, l1, a, b, ac):
    """
    Solve (aS+b*A+cB)x = f, where S, A and B are 4th order Laplace, stiffness and mass matrices of Shen with Dirichlet BC
    """
    y = np.zeros(u0.shape[0], dtype=fk.dtype)
    M = u0.shape[0]
    ForwardBsolve_L(y, l0, l1, fk)

    # Solve Backward U u = y
    BackBsolve_U(M, odd, y, uk, u0, u1, u2, l0, l1, a, b, ac)

@nb.jit(nopython=True, fastmath=True, cache=True)
def ForwardBsolve_L(y, l0, l1, fk):
    # Solve Forward Ly = f
    y[0] = fk[0]
    y[1] = fk[1] - l0[0]*y[0]
    N = l0.shape[0]
    for i in range(2, N):
        y[i] = fk[i] - l0[i-1]*y[i-1] - l1[i-2]*y[i-2]

@nb.jit(nopython=True, fastmath=True, cache=True)
def Solve_Biharmonic_2D(axis, b, u, u0, u1, u2, l0, l1, ak, bk, a0):
    if axis == 0:
        for j in range(b.shape[1]):
            Solve_Biharmonic_1D(b[:, j], u[:, j], u0[:, :, j], u1[:, :, j],
                                u2[:, :, j], l0[:, :, j], l1[:, :, j],
                                ak[:, :, j], bk[:, :, j], a0)
    elif axis == 1:
        for i in range(b.shape[0]):
            Solve_Biharmonic_1D(b[i], u[i], u0[:, i], u1[:, i],
                                u2[:, i], l0[:, i], l1[:, i],
                                ak[:, i], bk[:, i], a0)

@nb.jit(nopython=True, fastmath=True, cache=True)
def Solve_Biharmonic_3D(axis, b, u, u0, u1, u2, l0, l1, ak, bk, a0):
    if axis == 0:
        for j in range(b.shape[1]):
            for k in range(b.shape[2]):
                Solve_Biharmonic_1D(b[:, j, k], u[:, j, k], u0[:, :, j, k], u1[:, :, j, k],
                                    u2[:, :, j, k], l0[:, :, j, k], l1[:, :, j, k],
                                    ak[:, :, j, k], bk[:, :, j, k], a0)
    elif axis == 1:
       for i in range(b.shape[0]):
            for k in range(b.shape[2]):
                Solve_Biharmonic_1D(b[i, :, k], u[i, :, k], u0[:, i, :, k], u1[:, i, :, k],
                                    u2[:, i, :, k], l0[:, i, :, k], l1[:, i, :, k],
                                    ak[:, i, :, k], bk[:, i, :, k], a0)
    elif axis == 2:
       for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                Solve_Biharmonic_1D(b[i, j], u[i, j], u0[:, i, j], u1[:, i, j],
                                    u2[:, i, j], l0[:, i, j], l1[:, i, j],
                                    ak[:, i, j], bk[:, i, j], a0)

@nb.jit(nopython=True, fastmath=True, cache=True)
def Biharmonic_matvec2D(v, b, a0, alfa, beta,
                        sii, siu, siuu,
                        ail, aii, aiu,
                        bill, bil, bii, biu, biuu, axis):
    if axis == 0:
        for j in range(v.shape[1]):
            Biharmonic_matvec1D(v[:, j], b[:, j], a0, alfa[0, j],
                                beta[0, j], sii, siu, siuu,
                                ail, aii, aiu,
                                bill, bil, bii, biu, biuu)
    elif axis == 1:
        for i in range(v.shape[0]):
            Biharmonic_matvec1D(v[i], b[i], a0, alfa[i, 0],
                                beta[i, 0], sii, siu, siuu,
                                ail, aii, aiu,
                                bill, bil, bii, biu, biuu)

@nb.jit(nopython=True, fastmath=True, cache=True)
def Biharmonic_matvec3D(v, b, a0, alfa, beta,
                        sii, siu, siuu,
                        ail, aii, aiu,
                        bill, bil, bii, biu, biuu, axis):
    if axis == 0:
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                Biharmonic_matvec1D(v[:, j, k], b[:, j, k], a0, alfa[0, j, k],
                                    beta[0, j, k], sii, siu, siuu,
                                    ail, aii, aiu, bill, bil, bii, biu, biuu)
    elif axis == 1:
        for i in range(v.shape[0]):
            for k in range(v.shape[2]):
                Biharmonic_matvec1D(v[i, :, k], b[i, :, k], a0, alfa[i, 0, k],
                                    beta[i, 0, k], sii, siu, siuu,
                                    ail, aii, aiu, bill, bil, bii, biu, biuu)
    elif axis == 2:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                Biharmonic_matvec1D(v[i, j], b[i, j], a0, alfa[i, j, 0],
                                    beta[i, j, 0], sii, siu, siuu,
                                    ail, aii, aiu, bill, bil, bii, biu, biuu)


@nb.jit(nopython=True, fastmath=True, cache=True)
def Biharmonic_matvec1D(v, b, a0, alfa, beta,
                        sii, siu, siuu,
                        ail, aii, aiu,
                        bill, bil, bii, biu, biuu):
    N = sii.shape[0]
    ldd = np.empty(N)
    ld = np.empty(N)
    dd = np.empty(N)
    ud = np.empty(N)
    udd = np.empty(N)

    for i in range(N):
        dd[i] = a0*sii[i] + alfa*aii[i] + beta*bii[i]

    for i in range(N-2):
        ld[i] = alfa*ail[i] + beta*bil[i]

    for i in range(N-4):
        ldd[i] = beta*bill[i]

    for i in range(N-2):
        ud[i] = a0*siu[i] + alfa*aiu[i] + beta*biu[i]

    for i in range(N-4):
        udd[i] = a0*siuu[i] + beta*biuu[i]

    i = N-1
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i]
    i = N-2
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i]
    i = N-3
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i] + ud[i]*v[i+2]
    i = N-4
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i] + ud[i]*v[i+2]
    i = N-5
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i] + ud[i]*v[i+2] + udd[i]*v[i+4]
    i = N-6
    b[i] = ldd[i-4]*v[i-4]+ ld[i-2]* v[i-2] + dd[i]*v[i] + ud[i]*v[i+2] + udd[i]*v[i+4]

    s1 = 0.0
    s2 = 0.0
    o1 = 0.0
    o2 = 0.0
    for k in range(N-7, -1, -1):
        j = k+6
        p = k*sii[k]/(k+1.)
        r = 24*(k+1)*(k+2)*np.pi
        d = v[j]/(j+3.)
        if k % 2 == 0:
            s1 += d
            s2 += (j+2)*(j+2)*d
            b[k] = (p*s1 + r*s2)*a0
        else:
            o1 += d
            o2 += (j+2)*(j+2)*d
            b[k] = (p*o1 + r*o2)*a0

        if k > 3:
            b[k] += ldd[k-4]*v[k-4]+ ld[k-2]* v[k-2] + dd[k]*v[k] + ud[k]*v[k+2] + udd[k]*v[k+4]
        elif k > 1:
            b[k] += ld[k-2]* v[k-2] + dd[k]*v[k] + ud[k]*v[k+2] + udd[k]*v[k+4]
        else:
            b[k] += dd[k]*v[k] + ud[k]*v[k+2] + udd[k]*v[k+4]

def Biharmonic_matvec(v, b, a0, alfa, beta,
                      sii, siu, siuu, ail, aii, aiu,
                      bill, bil, bii, biu, biuu, axis):
    n = v.ndim
    if n == 1:
        Biharmonic_matvec1D(v, b, a0, alfa, beta,
                            sii, siu, siuu, ail, aii, aiu,
                            bill, bil, bii, biu, biuu)
    elif n == 2:
        Biharmonic_matvec2D(v, b, a0, alfa, beta,
                            sii, siu, siuu, ail, aii, aiu,
                            bill, bil, bii, biu, biuu, axis)
    elif n == 3:
        Biharmonic_matvec3D(v, b, a0, alfa, beta,
                            sii, siu, siuu, ail, aii, aiu,
                            bill, bil, bii, biu, biuu, axis)
