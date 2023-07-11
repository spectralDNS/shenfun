#cython: boundscheck=False
#cython: wraparound=False
#cython: language_level=3

from libc.math cimport sin, cos, exp, sqrt, lgamma, M_PI, M_2_PI
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
import array
import numpy as np
cimport cython
cimport numpy as np
import cython
from scipy.special import gammaln
np.import_array()

ctypedef fused T:
    double
    complex

ctypedef void (*funv)(T* const, T*, int, int , void* const)

ctypedef bint bool

cpdef double normf(double[::1] u):
    cdef:
        int i
        int N = u.shape[0]
        double s = 0
    for i in range(N):
        s += u[i]*u[i]
    s = sqrt(s)
    return s

cdef double norm(double* u, int N):
    cdef:
        int i
        double s = 0
    for i in range(N):
        s += u[i]*u[i]
    s = sqrt(s)
    return s

cdef double sum(double* u, int N):
    cdef:
        int i
        double s = 0
    for i in range(N):
        s += u[i]
    return s

cdef void IterAllButAxis(funv f, np.ndarray[T, ndim=1] input_array, np.ndarray[T, ndim=1] output_array, int st, int N, int axis, tuple shapein, tuple shapeout, void* const data):
    cdef:
        np.flatiter ita = np.PyArray_IterAllButAxis(np.PyArray_Reshape(input_array, shapein), &axis)
        np.flatiter ito = np.PyArray_IterAllButAxis(np.PyArray_Reshape(output_array, shapeout), &axis)

    while np.PyArray_ITER_NOTDONE(ita):
        f(<T* const>np.PyArray_ITER_DATA(ita), <T*>np.PyArray_ITER_DATA(ito), st, N, data)
        np.PyArray_ITER_NEXT(ita)
        np.PyArray_ITER_NEXT(ito)

cdef void _matvec(double* A, T* x, T* b, int m, int n, int transpose):
    cdef:
        int i, j, k
        T s
        double* a
    if transpose == 0:
        for i in range(m):
            s = 0.0
            a = &A[n*i]
            for j in range(n):
                s += a[j]*x[j]
            b[i] = s
    else:
        for j in range(n):
            b[j] = 0
        for i in range(m):
            s = x[i]
            a = &A[n*i]
            for j in range(n):
                b[j] += s*a[j]

cdef void _matvecadd(double* A, T* x, T* b, int m, int n, int transpose):
    cdef:
        int i, j, k
        T s
        double* a
    if transpose == 0:
        for i in range(m):
            s = 0.0
            a = &A[n*i]
            for j in range(n):
                s += a[j]*x[j]
            b[i] += s
    else:
        for i in range(m):
            s = x[i]
            a = &A[n*i]
            for j in range(n):
                b[j] += s*a[j]


cdef void _matvectri(double* A, T* x, T* b, int m, int n, int lda, int lower):
    cdef:
        T* xp
        double* ap
        T s
        int i, j

    if lower == 0:
        for i in range(m):
            s = 0.0
            xp = &x[0]
            ap = &A[i*lda]
            for j in range(i+1):
                s += ap[j]*xp[j]
            b[i] += s
    else:
        for i in range(m):
            s = 0.0
            xp = &x[i]
            ap = &A[i*lda+i]
            for j in range(n-i):
                s += ap[j]*xp[j]
            b[i] += s

ctypedef struct R0:
    double* a
    double* x
    double* Lnm
    double* Ln
    double* Lnp
    int No
    int M
    int Mx

cpdef np.ndarray restricted_product(L, np.ndarray input_array, np.ndarray output_array, double[::1] x, int i0, int i1, int a0, int axis, np.ndarray[double, ndim=2] a):
    cdef:
        int n = input_array.ndim
        int st, N
        int M = output_array.shape[axis]
        str dtype = input_array.dtype.char
        tuple shapein
        tuple shapeout = np.shape(output_array)
        np.ndarray[double, ndim=1] Lnm = L.evaluate_basis(x[i0:i1], i=a0)
        np.ndarray[double, ndim=1] Ln = L.evaluate_basis(x[i0:i1], i=a0+1)
        np.ndarray[double, ndim=1] Lnp = L.evaluate_basis(x[i0:i1], i=a0+2)
        np.ndarray[double, ndim=2] ax = a[:, slice(a0, None)].copy()
        R0 r0 = R0(&ax[0, 0],
                   &x[i0], &Lnm[0], &Ln[0], &Lnp[0],
                   M, ax.shape[0], ax.shape[1])
    sl = [slice(None)]*n
    sl[axis] = slice(i0, i1)
    input = input_array[tuple(sl)]
    if input.flags['C_CONTIGUOUS'] is False:
        input = input.copy()
    shapein = np.shape(input)
    N = input.shape[axis]
    st = input.strides[axis]//input.itemsize
    if dtype == 'd':
        IterAllButAxis[double](_restricted_product_ptr, np.PyArray_Ravel(input, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shapein, shapeout, &r0)
    elif dtype == 'D':
        IterAllButAxis[complex](_restricted_product_ptr, np.PyArray_Ravel(input, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shapein, shapeout, &r0)
    else:
        raise NotImplementedError
    return output_array

ctypedef struct S0:
    double* xj
    double* wj
    double* a
    int Nx
    int M
    int Mx

cpdef scalar_product(np.ndarray input_array, np.ndarray output_array, double[::1] x, double[::1] w, int axis, double[:, ::1] a):
    cdef:
        int n = input_array.ndim
        int st = input_array.strides[axis]//input_array.itemsize
        int N = input_array.shape[axis]
        str dtype = input_array.dtype.char
        tuple shapein = np.shape(input_array)
        tuple shapeout = np.shape(output_array)
        S0 s0 = S0(&x[0], &w[0], &a[0, 0], output_array.shape[axis], a.shape[0], a.shape[1])

    if dtype == 'd':
        IterAllButAxis[double](_scalar_product_ptr, np.PyArray_Ravel(input_array, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shapein, shapeout, &s0)
    elif dtype == 'D':
        IterAllButAxis[complex](_scalar_product_ptr, np.PyArray_Ravel(input_array, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shapein, shapeout, &s0)
    else:
        raise NotImplementedError

ctypedef struct E0:
    double* x
    double* a
    int Nx
    int M
    int Mx

cpdef evaluate_expansion_all(np.ndarray input_array, np.ndarray output_array, double[::1] x, int axis, double[:, ::1] a):
    cdef:
        int n = input_array.ndim
        int st = input_array.strides[axis]//input_array.itemsize
        int N = input_array.shape[axis]
        str dtype = input_array.dtype.char
        tuple shapein = np.shape(input_array)
        tuple shapeout = np.shape(output_array)
        E0 e0 = E0(&x[0], &a[0, 0], output_array.shape[axis], a.shape[0], a.shape[1])

    if not input_array.flags['C_CONTIGUOUS']:
        input_array = input_array.copy()
    if dtype == 'd':
        IterAllButAxis[double](_evaluate_expansion_all_ptr, np.PyArray_Ravel(input_array, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shapein, shapeout, &e0)
    elif dtype == 'D':
        IterAllButAxis[complex](_evaluate_expansion_all_ptr, np.PyArray_Ravel(input_array, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shapein, shapeout, &e0)
    else:
        raise NotImplementedError

@cython.cdivision(True)
cdef void _evaluate_expansion_all_ptr(T* const ui,
                                      T* uo,
                                      int st,
                                      int N,
                                      void* const data):
    cdef:
        int i, j
        T s
        E0* e0 = <E0*> data
        int Nx = e0.Nx
        int Mx = e0.Mx
        int M = e0.M
        double s1, s2, a00
        double* xj = e0.x
        double* a = e0.a
        double* anm = &a[0]
        double* anp
        double* ann
        double *an = <double*>malloc(Mx*sizeof(double))
        double *Lnm = <double*>malloc(Nx*sizeof(double))
        double *Ln = <double*>malloc(Nx*sizeof(double))
        double *Lnp = <double*>malloc(Nx*sizeof(double))

    if M == 2:
        for i in range(Mx):
            an[i] = 0
        ann = &an[0]
        anp = &a[Mx]
    else:
        ann = &a[Mx]
        anp = &a[2*Mx]
    for i in range(Nx):
        Lnm[i] = 1
        Ln[i] = (xj[i]-ann[0])/anm[0]
        uo[i*st] = 0
        Lnp[i] = (xj[i]-ann[1])/anm[1]*Ln[i] - anp[1]/anm[1]*Lnm[i]
    for i in range(N):
        s1 = 1/anm[i+2]
        s2 = anp[i+2]/anm[i+2]
        a00 = ann[i+2]
        for j in range(Nx):
            uo[j*st] += Lnm[j]*ui[i*st]
            Lnm[j] = Ln[j]
            Ln[j] = Lnp[j]
            Lnp[j] = s1*(xj[j]-a00)*Ln[j] - s2*Lnm[j]
    free(an)
    free(Lnm)
    free(Ln)
    free(Lnp)

@cython.cdivision(True)
cdef void _scalar_product_ptr(T* const ui,
                              T* uo,
                              int st,
                              int N,
                              void* const data):
    cdef:
        int i, j
        T s
        double s1, s2, a00
        S0* s0 = <S0*> data
        int M = s0.M
        int Mx = s0.Mx
        int Nx = s0.Nx
        double* xj = s0.xj
        double* wj = s0.wj
        double* a = s0.a
        double* anm = &a[0]
        double* ann
        double* anp
        double *an = <double*>malloc(Mx*sizeof(double))
        double *Lnm = <double*>malloc(Nx*sizeof(double))
        double *Ln = <double*>malloc(Nx*sizeof(double))
        double *Lnp = <double*>malloc(Nx*sizeof(double))

    if M == 2:
        for i in range(Mx):
            an[i] = 0
        ann = &an[0]
        anp = &a[Mx]
    else:
        ann = &a[Mx]
        anp = &a[2*Mx]

    for i in range(Nx):
        Lnm[i] = 1
        Ln[i] = (xj[i]-ann[0])/anm[0]
        uo[i*st] = 0
        Lnp[i] = (xj[i]-ann[1])/anm[1]*Ln[i] - anp[1]/anm[1]*Lnm[i]

    for i in range(N):
        s1 = 1/anm[i+2]
        s2 = anp[i+2]/anm[i+2]
        a00 = ann[i+2]
        s = 0.0
        for j in range(Nx):
            s += Lnm[j]*wj[j]*ui[j*st]
            Lnm[j] = Ln[j]
            Ln[j] = Lnp[j]
            Lnp[j] = s1*(xj[j]-a00)*Ln[j] - s2*Lnm[j]
        uo[i*st] = s
    free(an)
    free(Lnm)
    free(Ln)
    free(Lnp)


@cython.cdivision(True)
cdef void _restricted_product_ptr(T* const input_array,
                                  T* output_array,
                                  int st,
                                  int N,
                                  void* const data):

    cdef:
        int k, kp, i
        double a00
        T s
        R0* r0 = <R0*>data
        double* xi = r0.x
        double* a = r0.a
        int No = r0.No
        int M = r0.M
        int Mx = r0.Mx
        double* anm = &a[0]
        double* ann
        double* anp
        double *an = <double*>malloc(Mx*sizeof(double))
        double *Lnm = <double*>malloc(N*sizeof(double))
        double *Ln = <double*>malloc(N*sizeof(double))
        double *Lnp = <double*>malloc(N*sizeof(double))

    if M == 2:
        for i in range(Mx):
            an[i] = 0
        ann = &an[0]
        anp = &a[Mx]
    else:
        ann = &a[Mx]
        anp = &a[2*Mx]
    for i in range(N):
        Lnm[i] = r0.Lnm[i]
        Ln[i] = r0.Ln[i]
        Lnp[i] = r0.Lnp[i]
    for k in range(No):
        s1 = 1/anm[k+2]
        s2 = anp[k+2]/anm[k+2]
        a00 = ann[k+2]
        s = 0.0
        for i in range(N):
            s += Lnm[i]*input_array[i*st]
            Lnm[i] = Ln[i]
            Ln[i] = Lnp[i]
            Lnp[i] = s1*(xi[i]-a00)*Ln[i] - s2*Lnm[i]
        output_array[k*st] = s
    free(an)
    free(Lnm)
    free(Ln)
    free(Lnp)

cdef int get_h(int level, long* D, int N):
    cdef:
        int i
        int h = 1
    if level < N-1:
        for i in range(level+1, N):
            h *= D[i]
    return h

cdef int get_number_of_blocks(int level, long* D, int N):
    cdef:
        int s = 1
        int nd = 1
        int i, j
    if level > 0:
        for i in range(level):
            nd = 1
            for j in range(i+1):
                nd *= D[level-j-1]
            s += nd
    return s

cdef int get_number_of_submatrices(long* D, int L):
    cdef:
        int Ns, level
    Ns = 0
    for level in range(L):
        Ns += get_number_of_blocks(level, D, L)*D[level]*(D[level]+1)//2
    return Ns

cdef void get_ij(int* ij, int level, int block, int diags, int h, int s, long* D, int L):
    cdef:
        int i, i0, j0
    ij[0] = block*D[level]*s*get_h(level, D, L)
    ij[1] = diags+ij[0]
    for i in range(level+1, L):
        ij[1] += s*get_h(i, D, L)

cdef double chebvalS(double *x, double* c, int M):
    cdef:
        double x2, c0, c1, tmp
        int i
    x2 = 2*x[0]
    c0 = c[M-2]
    c1 = c[M-1]
    for i in range(3, M + 1):
        tmp = c0
        c0 = c[M-i] - c1
        c1 = tmp + c1*x2
    return c0+c1*x[0]

cdef void chebvalC(double* x, int N, double* c, int M, double* c0):
    cdef:
        double* x2 = <double*>malloc(N*sizeof(double))
        double* c1 = <double*>malloc(N*sizeof(double))
        int i, j
        double tmp

    for j in range(N):
        x2[j] = 2*x[j]
        c0[j] = c[M-2]
        c1[j] = c[M-1]
    for i in range(3, M + 1):
        for j in range(N):
            tmp = c0[j]
            c0[j] = c[M-i] - c1[j]
            c1[j] = tmp + c1[j]*x2[j]
    for j in range(N):
        c0[j] += c1[j]*x[j]
    free(x2)
    free(c1)

cpdef np.ndarray[double, ndim=1] chebval(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] u):
    cdef:
        int N = x.shape[0]
        int M = u.shape[0]
        np.ndarray[double, ndim=1] c = u.copy()
        np.ndarray[double, ndim=1] x2 = x.copy()
        np.ndarray[double, ndim=1] c0 = np.zeros_like(x)
        np.ndarray[double, ndim=1] c1 = np.zeros_like(x)
        int i, j
        double tmp

    if M == 1:
        c0[:] = c[0]
        c1[:] = 0
    elif M == 2:
        c0[:] = c[0]
        c1[:] = c[1]
    else:
        for i in range(N):
            x2[i] = 2*x[i]
        c0[:] = c[M-2]
        c1[:] = c[M-1]
        for i in range(3, M + 1):
            for j in range(N):
                tmp = c0[j]
                c0[j] = c[M-i] - c1[j]
                c1[j] = tmp + c1[j]*x2[j]
    for j in range(N):
        c0[j] += c1[j]*x[j]
    return c0

def Lambda(np.ndarray x):
    """Return

    .. math::

            \Lambda(x) = \frac{\Gamma(x+\frac{1}{2})}{\Gamma(x+1)}

    Parameters
    ----------
    x : array of floats
    """
    cdef:
        int ndim = np.PyArray_NDIM(x)
        np.ndarray[double, ndim=1] a = _Lambda(np.PyArray_Ravel(x, np.NPY_CORDER))
    return np.PyArray_Reshape(a, np.shape(x))

@cython.cdivision(True)
cpdef np.ndarray[double, ndim=1] _Lambda(double[::1] z):
    cdef:
        int N = z.shape[0]
        int i = 0
        double x
        np.ndarray[double, ndim=1] a = np.empty(N)
        double[8] a0 = [9.9688251374224490e-01,
                        -3.1149502185860763e-03,
                        2.5548605043159494e-06,
                        1.8781800445057731e-08,
                        -4.0437919461099256e-11,
                        -9.0003384202201278e-13,
                        3.1032782098712292e-15,
                        1.0511830721865363e-16]

    for i in range(N):
        if z[i] > 20:
            x = -1+40/z[i]
            a[i] = chebvalS(&x, a0, 8)
            a[i] /= sqrt(z[i])
        else:
            a[i] = exp(lgamma(z[i]+0.5) - lgamma(z[i]+1))
    return a

ctypedef struct DL:
    double* a
    int transpose

@cython.cdivision(True)
cpdef _leg2cheb(np.ndarray[T, ndim=1] ui, np.ndarray[T, ndim=1] uo, double[::1] a, int trans):
    cdef:
        DL d = DL(&a[0], trans)
        int st = ui.strides[0]
        int sz = <int>ui.itemsize
        int N = ui.shape[0]
    _leg2cheb_ptr[T](&ui[0], &uo[0], st//sz, N, &d)

ctypedef struct DI:
    double* a
    double* d

cpdef _cheb2leg(T[::1] ui, T[::1] uo, double[::1] dn, double[::1] a):
    cdef DI di = DI(&a[0], &dn[0])
    _cheb2leg_ptr[T](&ui[0], &uo[0], ui.strides[0]/ui.itemsize, ui.shape[0], &di)

cpdef np.ndarray cheb2leg(np.ndarray input_array, np.ndarray output_array, int axis=0):
    cdef:
        int N = input_array.shape[axis]
        int st = input_array.strides[axis]//input_array.itemsize
        int Ne = N//2+N%2
        str dtype = input_array.dtype.char
        np.ndarray[double, ndim=1] k = np.arange(N, dtype='d')
        np.ndarray[double, ndim=1] a = np.zeros(N)
        np.ndarray[double, ndim=1] dn = np.zeros(Ne)
        DI d = DI(&a[0], &dn[0])
        tuple shape = np.shape(input_array)

    k[0] = 1
    dn[:] = _Lambda((k[::2]-2)/2)/k[::2]
    a[:] = 1/(2*_Lambda(k)*k*(k+0.5))
    a[0] = 2/np.sqrt(np.pi)
    output_array.fill(0)
    if dtype == 'd':
        IterAllButAxis[double](_cheb2leg_ptr, np.PyArray_Ravel(input_array, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shape, shape, &d)
    elif dtype == 'D':
        IterAllButAxis[complex](_cheb2leg_ptr, np.PyArray_Ravel(input_array, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shape, shape, &d)
    else:
        raise NotImplementedError
    return output_array

cpdef np.ndarray leg2cheb(np.ndarray input_array, np.ndarray output_array, int axis=0, int transpose=0):
    cdef:
        int N = input_array.shape[axis]
        int st = input_array.strides[axis]//input_array.itemsize
        str dtype = input_array.dtype.char
        np.ndarray[double, ndim=1] a = Lambda(np.arange(N, dtype=np.float64))
        DL d = DL(&a[0], transpose)
        tuple shape = np.shape(input_array)

    output_array.fill(0)
    if dtype == 'd':
        IterAllButAxis[double](_leg2cheb_ptr, np.PyArray_Ravel(input_array, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shape, shape, &d)
    elif dtype == 'D':
        IterAllButAxis[complex](_leg2cheb_ptr, np.PyArray_Ravel(input_array, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shape, shape, &d)
    else:
        raise NotImplementedError
    return output_array

@cython.cdivision(True)
cdef void _leg2cheb_ptr(T* const c,
                        T* v,
                        int st,
                        int N,
                        void* const data):
    cdef:
        int n, i, ii
        DL* d = <DL*>data
        double* a = d.a
        double* ap
        T* cp
        int transpose = d.transpose
        T *cv = <T *> malloc(N*transpose*sizeof(T))

    if transpose == 0:
        for n in range(0, N, 2):
            ap = &a[n//2]
            cp = &c[n*st]
            ii = 0
            for i in range(0, N-n):
                v[ii] += ap[0]*ap[i]*cp[ii]
                ii += st
        v[0] *= 0.5
        ii = 0
        for n in range(0, N):
            v[ii] *= M_2_PI
            ii += st
    else:
        ii = 0
        for i in range(0, N):
            cv[i] = M_2_PI*c[ii]
            ii += st
        cv[0] *= 0.5
        for n in range(0, N, 2):
            ap = &a[n//2]
            ii = n*st
            for i in range(0, N-n):
                v[ii] += ap[0]*ap[i]*cv[i]
                ii += st
    free(cv)

@cython.cdivision(True)
cdef void _cheb2leg_ptr(T* const v,
                        T* c,
                        int st,
                        int N,
                        void* const data):
    cdef:
        int n, i, ii
        double SPI = sqrt(M_PI)
        DI* di = <DI*>data
        double* a = di.a
        double* d = di.d
        T* vn = <T*>malloc(N*sizeof(T))
    vn[0] = v[0]
    ii = st
    for i in range(1, N):
        vn[i] = v[ii]*i
        ii += st
    ii = 0
    for n in range(N):
        c[ii] = SPI*a[n]*vn[n]
        ii += st
    for n in range(2, N, 2):
        ii = 0
        for i in range(0, N-n):
            c[ii] -= d[n//2]*a[n//2+i]*vn[n+i]
            ii += st
    ii = 0
    for n in range(N):
        c[ii] *= (n+0.5)
        ii += st
    free(vn)

ctypedef struct DCN:
    int Nn
    double* A
    double* T
    double* Th
    double* ThT
    long* Nk
    long* D
    long* Mmin
    long* uD
    long* cuD
    size_t Lh
    size_t L
    size_t s
    size_t diags
    bool trans

cdef size_t find_index(long* v, size_t u, size_t N):
    cdef size_t i = 1
    if v[0] == u:
        return 0
    while v[i] != u:
        i += 1
    return i

@cython.cdivision(True)
cpdef void FMMcheb(np.ndarray input_array, np.ndarray output_array, int axis, int Nn, double[::1] A, np.ndarray[long, ndim=1] Nk, double[:, :, ::1] T, double[:, :, ::1] Th, double[:, :, ::1] ThT, long[::1] D, long[::1] Mmin, int s, int diags, int trans):
    cdef:
        np.ndarray[long, ndim=1] uD = np.hstack((0, np.unique(D)))
        np.ndarray[long, ndim=1] cuD = np.cumsum(uD)
        DCN dc = DCN(Nn, &A[0], &T[0, 0, 0], &Th[0, 0, 0], &ThT[0, 0, 0], &Nk[0], &D[0], &Mmin[0], &uD[0], &cuD[0], len(uD), len(D), s, diags, trans)
        int st = input_array.strides[axis]//input_array.itemsize
        int N = input_array.shape[axis]
        dtype = input_array.dtype.char
        tuple shape = np.shape(input_array)
    if dtype == 'd':
        IterAllButAxis[double](_FMMcheb_ptr, np.PyArray_Ravel(input_array, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shape, shape, &dc)
    elif dtype == 'D':
        IterAllButAxis[complex](_FMMcheb_ptr, np.PyArray_Ravel(input_array, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shape, shape, &dc)
    else:
        raise NotImplementedError

@cython.cdivision(True)
cdef void _divmod(size_t a, size_t b, size_t* i0, size_t* j0):
    i0[0] = a // b;
    j0[0] = a % b;

@cython.cdivision(True)
cdef void _FMMcheb_ptr(T* u,
                       T* v,
                       int st,
                       int N,
                       void* data):
    cdef:
        size_t ik, i0, j0, j1, i, M, odd, Nc
        size_t s0, s1, M0, Mmax, block, level, p, q, h, b0, p0, q0
        int ij[2]
        DCN* dc = <DCN*>data
        double* A = dc.A
        double* TT = dc.T
        long* Nk = dc.Nk
        long* D = dc.D
        long* Mmin = dc.Mmin
        double* Th = dc.Th
        double* ThT = dc.ThT
        long* uD = dc.uD
        long* cuD = dc.cuD
        size_t Lh = dc.Lh
        size_t Nn = dc.Nn
        size_t L = dc.L
        size_t s = dc.s
        size_t diags = dc.diags
        bool trans = dc.trans
        T* cia = <T*>malloc(Nn//2*sizeof(T))
        T* coa = <T*>malloc(Nn//2*sizeof(T))
        T** wk = <T**>malloc(L*sizeof(T*))
        T** ck = <T**>malloc(L*sizeof(T*))
        T* c0
        T* c1
        T* w0
        T* w1

    Mmax = 0
    for i in range(L):
        Mmax = max(Mmin[i], Mmax)

    for level in range(L):
        ck[level] = <T*>malloc(get_number_of_blocks(level, D, L)*D[level]*Mmax*sizeof(T))
        wk[level] = <T*>malloc(get_number_of_blocks(level, D, L)*D[level]*Mmax*sizeof(T))
    jj = 0 if level == L-1 else D[level+1]
    if trans == 0:
        for odd in range(2):
            for i in range(Nn//2):
                cia[i] = 0
                coa[i] = 0
            for i in range(N//2+(N%2)*(1-odd)):
                cia[i] = u[(2*i+odd)*st]
            for level in range(L):
                for i in range(get_number_of_blocks(level, D, L)*D[level]*Mmax):
                    wk[level][i] = 0.0
                    ck[level][i] = 0.0
            ik = 0
            Nc = 0
            for level in range(L-1, -1, -1):
                M0 = Mmin[level]
                h = s*get_h(level, D, L)
                s0 = find_index(uD, D[level], Lh)-1
                s1 = cuD[s0]

                for block in range(get_number_of_blocks(level, D, L)):
                    get_ij(&ij[0], level, block, diags, h, s, D, L)
                    c0 = &ck[level][block*D[level]*Mmax]
                    w0 = &wk[level][block*D[level]*Mmax]

                    for q in range(D[level]):
                        if level == L-1:
                            _matvec(&TT[odd*s*M0], &cia[ij[1]+(q+1)*s], &w0[q*Mmax], s, M0, 1)
                        if level > 0 and block > 0:
                            _divmod(block-1, D[level-1], &b0, &q0)
                            _matvectri(&Th[(s1+q)*Mmax*Mmax], &w0[q*Mmax], &wk[level-1][(b0*D[level-1]+q0)*Mmax], M0, M0, Mmax, 0)

                        for p in range(q+1):
                            M = Nk[ik]
                            _matvecadd(&A[Nc], &w0[q*Mmax], &c0[p*Mmax], M, M, 0)
                            Nc += M*M
                            ik += 1

            for level in range(L-1):
                c0 = ck[level]
                c1 = ck[level+1]
                M0 = Mmin[level]
                j1 = 0
                s0 = find_index(uD, D[level+1], Lh)-1
                s1 = cuD[s0]
                for block in range(get_number_of_blocks(level+1, D, L)-1):
                    for p in range(D[level+1]):
                        _matvectri(&ThT[(s1+p)*Mmax*Mmax], &c0[block*Mmax], &c1[j1*Mmax], M0, M0, Mmax, 1)
                        j1 += 1

            M0 = Mmin[L-1]
            level = L-1
            for block in range(get_number_of_blocks(level, D, L)):
                get_ij(&ij[0], level, block, diags, s, s, D, L)
                c0 = &ck[level][block*D[level]*Mmax]
                for p in range(D[level]):
                    _matvecadd(&TT[odd*s*M0], &c0[p*Mmax], &coa[ij[0]+p*s], s, M0, 0)

            for i in range(N//2+(N%2)*(1-odd)):
                v[(2*i+odd)*st] = coa[i]

    else:
        for odd in range(2):
            for i in range(Nn//2):
                cia[i] = 0
                coa[i] = 0
            for i in range(N//2+(N%2)*(1-odd)):
                cia[i] = u[(2*i+odd)*st]
            for level in range(L):
                for i in range(get_number_of_blocks(level, D, L)*D[level]*Mmax):
                    wk[level][i] = 0.0
                    ck[level][i] = 0.0
            ik = 0
            Nc = 0
            for level in range(L-1, -1, -1):
                M0 = Mmin[level]
                h = s*get_h(level, D, L)
                w0 = wk[level]
                for block in range(get_number_of_blocks(level, D, L)):
                    get_ij(&ij[0], level, block, diags, h, s, D, L)
                    i0 = ij[1]
                    j0 = ij[0]
                    c0 = &ck[level][block*D[level]*Mmax]
                    w0 = &wk[level][block*D[level]*Mmax]
                    s0 = find_index(uD, D[level], Lh)-1
                    s1 = cuD[s0]
                    _divmod(block, D[level], &b0, &q0)
                    for p in range(D[level]):
                        for q in range(p+1):
                            if q == p:
                                if level == L-1:
                                    _matvecadd(&TT[odd*s*M0], &cia[j0+q*s], &w0[q*Mmax], s, M0, 1)
                                if level > 0 and block < get_number_of_blocks(level, D, L)-1:
                                    _matvectri(&Th[(s1+q)*Mmax*Mmax], &w0[q*Mmax], &wk[level-1][(b0*D[level]+q0)*Mmax], M0, M0, Mmax, 0)
                            M = Nk[ik]
                            _matvecadd(&A[Nc], &w0[q*Mmax], &c0[p*Mmax], M, M, 1)
                            Nc += M*M
                            ik += 1

            for level in range(L-1):
                c0 = ck[level]
                c1 = ck[level+1]
                M0 = Mmin[level]
                s0 = find_index(uD, D[level+1], Lh)-1
                s1 = cuD[s0]
                for block in range(1, get_number_of_blocks(level+1, D, L)):
                    for p in range(D[level+1]):
                        _divmod(block-1, D[level], &b0, &p0)
                        _matvectri(&ThT[(s1+p)*Mmax*Mmax], &c0[(b0*D[level]+p0)*Mmax], &c1[(block*D[level+1]+p)*Mmax], M0, M0, Mmax, 1)

            M0 = Mmin[L-1]
            level = L-1
            for block in range(get_number_of_blocks(level, D, L)):
                get_ij(&ij[0], level, block, diags, s, s, D, L)
                i0 = ij[1]
                j0 = ij[0]
                c0 = &ck[level][block*D[level]*Mmax]
                for p in range(D[level]):
                    _matvecadd(&TT[odd*s*M0], &c0[p*Mmax], &coa[i0+(p+1)*s], s, M0, 0)

            for i in range(N//2+(N%2)*(1-odd)):
                v[(2*i+odd)*st] = coa[i]

    free(cia)
    free(coa)
    for level in range(L):
        free(wk[level])
        free(ck[level])
    free(wk)
    free(ck)


ctypedef struct D1:
    double* a
    int n0
    int trans


cpdef void FMMdirect1(np.ndarray input_array, np.ndarray output_array, int axis, double[::1] a, int n0, int trans):
    cdef:
        D1 d1 = D1(&a[0], n0, trans)
        int st = input_array.strides[axis]//input_array.itemsize
        int N = input_array.shape[axis]
        dtype = input_array.dtype.char
        tuple shape = np.shape(input_array)

    if dtype == 'd':
        IterAllButAxis[double](_FMMdirect1_ptr, np.PyArray_Ravel(input_array, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shape, shape, &d1)
    elif dtype == 'D':
        IterAllButAxis[complex](_FMMdirect1_ptr, np.PyArray_Ravel(input_array, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shape, shape, &d1)
    else:
        raise NotImplementedError

@cython.cdivision(True)
cdef void _FMMdirect1_ptr(T* const u,
                          T* v,
                          int st,
                          int N,
                          void* const data):
    cdef:
        D1* d1 = <D1*>data
        int n0 = d1.n0
        int trans = d1.trans
        double* a = d1.a
        double* ap
        T* cp
        int n, i, ii

    if trans == 0:
        for n in range(0, 2*n0, 2):
            ap = &a[n//2]
            cp = &u[n*st]
            ii = 0
            for i in range(0, N-n):
                v[ii] += ap[0]*ap[i]*cp[ii]
                ii += st
    else:
        for n in range(0, 2*n0, 2):
            ap = &a[n//2]
            cp = &v[n*st]
            ii = 0
            for i in range(0, N-n):
                cp[ii] += ap[0]*ap[i]*u[ii]
                ii += st

ctypedef struct D2:
    double* a
    int h
    int Nd
    int n0
    int trans

cpdef void FMMdirect2(np.ndarray input_array, np.ndarray output_array, int axis, double[::1] a, int h, int Nd, int n0, int trans):
    cdef:
        D2 d2 = D2(&a[0], h, Nd, n0, trans)
        int st = input_array.strides[axis]//input_array.itemsize
        int N = input_array.shape[axis]
        dtype = input_array.dtype.char
        tuple shape= np.shape(input_array)

    if dtype == 'd':
        IterAllButAxis[double](_FMMdirect2_ptr, np.PyArray_Ravel(input_array, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shape, shape, &d2)
    elif dtype == 'D':
        IterAllButAxis[complex](_FMMdirect2_ptr, np.PyArray_Ravel(input_array, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shape, shape, &d2)
    else:
        raise NotImplementedError

@cython.cdivision(True)
cdef void _FMMdirect2_ptr(T* const u,
                          T* v,
                          int st,
                          int N,
                          void* const data):
    cdef:
        int n, i, j, k, d, i0, j0, n1, ii, Nm
        D2* d2 = <D2*>data
        double* a = d2.a
        int h = d2.h
        int Nd = d2.Nd
        int n0 = d2.n0
        int trans = d2.trans
        T* vp
        T* up
        double* ap

    for k in range(Nd):
        i0 = k*h
        j0 = n0+i0
        for n in range(0, h, 2):
            ap = &a[i0+(n0+n)//2]
            Nm = min(N-(j0+n), h-n)
            n1 = (n+n0)//2
            if trans == 0:
                vp = &v[i0*st]
                up = &u[(j0+n)*st]
                ii = 0
                for j in range(Nm):
                    vp[ii] += a[n1]*ap[j]*up[ii]
                    ii += st
            else:
                vp = &v[(j0+n)*st]
                up = &u[i0*st]
                ii = 0
                for j in range(Nm):
                    vp[ii] += a[n1]*ap[j]*up[ii]
                    ii += st


ctypedef struct D3:
    double* dn
    double* a
    int h
    int Nd
    int n0

cpdef void FMMdirect3(np.ndarray input_array, np.ndarray output_array, int axis, double[::1] dn, double[::1] a, int h, int Nd, int n0):
    cdef:
        int st = input_array.strides[axis]//input_array.itemsize
        int N = input_array.shape[axis]
        dtype = input_array.dtype.char
        D3 d3 = D3(&dn[0], &a[0], h, Nd, n0)
        tuple shape = np.shape(input_array)

    if dtype == 'd':
        IterAllButAxis[double](_FMMdirect3_ptr, np.PyArray_Ravel(input_array, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shape, shape, &d3)
    elif dtype == 'D':
        IterAllButAxis[complex](_FMMdirect3_ptr, np.PyArray_Ravel(input_array, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shape, shape, &d3)
    else:
        raise NotImplementedError


cdef void _FMMdirect3_ptr(T* const u,
                          T* v,
                          int st,
                          int N,
                          void* const data):
    cdef:
        int n, i, j, k, d, n1, i0, j0, ii, Nm
        D3* d3 = <D3*>data
        double* a = d3.a
        double* dn = d3.dn
        int h = d3.h
        int Nd = d3.Nd
        int n0 = d3.n0
        T* up
        T* vp
        double* dp
        double* ap

    for k in range(Nd):
        i0 = k*h
        j0 = n0+i0
        vp = &v[i0*st]
        for n in range(0, h, 2):
            n1 = (n+n0)//2
            Nm = min(N-(j0+n), h-n)
            ap = &a[i0+n1]
            up = &u[(j0+n)*st]
            ii = 0
            for j in range(Nm):
                vp[ii] -= dn[n1]*ap[j]*up[ii]
                ii += st

ctypedef struct D4:
    double* dn
    double* a
    int n0

cdef void _FMMdirect4_ptr(T* const u,
                          T* v,
                          int st,
                          int N,
                          void* const data):
    cdef:
        int n, i, ii
        D4* d4 = <D4*>data
        double* a = d4.a
        double* dn = d4.dn
        int n0 = d4.n0
        double* dp
        double* ap
        T* up
        double SPI = sqrt(M_PI)

    for i in range(N):
        v[i*st] += SPI*a[i]*u[i*st]
    for n in range(2, 2*n0, 2):
        ap = &a[n//2]
        up = &u[n*st]
        ii = 0
        for i in range(N-n):
            v[ii] -= dn[n//2]*ap[i]*up[ii]
            ii += st

cpdef void FMMdirect4(np.ndarray input_array, np.ndarray output_array, int axis, double[::1] dn, double[::1] a, int n0):
    cdef:
        int st = input_array.strides[axis]//input_array.itemsize
        int N = input_array.shape[axis]
        dtype = input_array.dtype.char
        D4 d4 = D4(&dn[0], &a[0], n0)
        tuple shape = np.shape(input_array)

    if dtype == 'd':
        IterAllButAxis[double](_FMMdirect4_ptr, np.PyArray_Ravel(input_array, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shape, shape, &d4)
    elif dtype == 'D':
        IterAllButAxis[complex](_FMMdirect4_ptr, np.PyArray_Ravel(input_array, np.NPY_CORDER), np.PyArray_Ravel(output_array, np.NPY_CORDER), st, N, axis, shape, shape, &d4)
    else:
        raise NotImplementedError
