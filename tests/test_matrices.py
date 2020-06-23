from copy import copy, deepcopy
import functools
from itertools import product
import numpy as np
import sympy as sp
from scipy.sparse.linalg import spsolve
from mpi4py import MPI
import pytest
import mpi4py_fft
import shenfun
from shenfun.chebyshev import matrices as cmatrices
from shenfun.chebyshev import bases as cbases
from shenfun.legendre import matrices as lmatrices
from shenfun.legendre import bases as lbases
from shenfun.laguerre import matrices as lagmatrices
from shenfun.laguerre import bases as lagbases
from shenfun.hermite import matrices as hmatrices
from shenfun.hermite import bases as hbases
from shenfun.jacobi import matrices as jmatrices
from shenfun.jacobi import bases as jbases
from shenfun.chebyshev import la as cla
from shenfun.legendre import la as lla
from shenfun import div, grad, inner, TensorProductSpace, FunctionSpace, SparseMatrix, \
    Function
from shenfun.spectralbase import inner_product

cBasis = (cbases.Orthogonal,
          cbases.ShenDirichletBasis,
          cbases.ShenNeumannBasis,
          cbases.ShenBiharmonicBasis)

# Bases with only GC quadrature
cBasisGC = (cbases.UpperDirichletBasis,
            cbases.ShenBiPolarBasis)

lBasis = (lbases.Orthogonal,
          lbases.ShenDirichletBasis,
          functools.partial(lbases.ShenDirichletBasis, scaled=True),
          lbases.ShenBiharmonicBasis,
          lbases.ShenNeumannBasis)

# Bases with only LG quadrature
lBasisLG = (lbases.UpperDirichletBasis,
            lbases.ShenBiPolarBasis,
            lbases.ShenBiPolar0Basis)

lagBasis = (lagbases.Orthogonal,
            lagbases.ShenDirichletBasis)

hBasis = (hbases.Orthogonal,)

jBasis = (jbases.Orthogonal,
          jbases.ShenDirichletBasis,
          jbases.ShenBiharmonicBasis,
          jbases.ShenOrder6Basis)

cquads = ('GC', 'GL')
lquads = ('LG', 'GL')
lagquads = ('LG',)
hquads = ('HG',)
jquads = ('JG',)
formats = ('dia', 'cython', 'python', 'self')

N = 12
k = np.arange(N).astype(float)
a = np.random.random(N)
c = np.zeros(N)
c1 = np.zeros(N)

work = {
    3:
    (np.random.random((N, N, N)),
     np.zeros((N, N, N)),
     np.zeros((N, N, N))),
    2:
    (np.random.random((N, N)),
     np.zeros((N, N)),
     np.zeros((N, N)))
    }

cbases2 = list(product(cBasis, cBasis))+list(product(cBasisGC, cBasisGC))
lbases2 = list(product(lBasis, lBasis))+list(product(lBasisLG, lBasisLG))
lagbases2 = list(product(lagBasis, lagBasis))
hbases2 = list(product(hBasis, hBasis))
jbases2 = list(product(jBasis, jBasis))
bases2 = cbases2+lbases2+lagbases2+hbases2+jbases2

cmats_and_quads = [list(k[0])+[k[1]] for k in product([(k, v) for k, v in cmatrices.mat.items()], cquads)]
lmats_and_quads = [list(k[0])+[k[1]] for k in product([(k, v) for k, v in lmatrices.mat.items()], lquads)]
lagmats_and_quads = [list(k[0])+[k[1]] for k in product([(k, v) for k, v in lagmatrices.mat.items()], lagquads)]
hmats_and_quads = [list(k[0])+[k[1]] for k in product([(k, v) for k, v in hmatrices.mat.items()], hquads)]
jmats_and_quads = [list(k[0])+[k[1]] for k in product([(k, v) for k, v in jmatrices.mat.items()], jquads)]
mats_and_quads = cmats_and_quads+lmats_and_quads+lagmats_and_quads+hmats_and_quads+jmats_and_quads

#cmats_and_quads_ids = ['-'.join(i) for i in product([v.__name__ for v in cmatrices.mat.values()], cquads)]
#lmats_and_quads_ids = ['-'.join(i) for i in product([v.__name__ for v in lmatrices.mat.values()], lquads)]

x = sp.symbols('x', real=True)
xp = sp.symbols('x', real=True, positive=True)

@pytest.mark.parametrize('key, mat, quad', mats_and_quads)
def test_mat(key, mat, quad):
    """Test that matrices built by hand equals those automatically generated"""
    test = key[0]
    trial = key[1]
    measure = 1
    if len(key) == 4:
        domain = key[2]
        measure = key[3]
        if quad == 'GL':
            return
    t0 = test[0]
    t1 = trial[0]
    if len(key) == 4:
        t0 = functools.partial(t0, domain=domain)
        t1 = functools.partial(t1, domain=domain)
    testfunction = (t0(N, quad=quad), test[1])
    trialfunction = (t1(N, quad=quad), trial[1])
    mat = mat(testfunction, trialfunction, measure=measure)
    shenfun.check_sanity(mat, testfunction, trialfunction, measure)
    if test[0].family() == 'Legendre' and test[0].boundary_condition() == 'Dirichlet':
        testfunction = (test[0](N, quad=quad, scaled=True), test[1])
        trialfunction = (trial[0](N, quad=quad, scaled=True), trial[1])
        mat = mat(testfunction, trialfunction)
        shenfun.check_sanity(mat, testfunction, trialfunction, measure)

@pytest.mark.parametrize('b0,b1', cbases2)
@pytest.mark.parametrize('quad', cquads)
@pytest.mark.parametrize('format', formats)
@pytest.mark.parametrize('dim', (2, 3))
@pytest.mark.parametrize('k', range(5))
def test_cmatvec(b0, b1, quad, format, dim, k):
    """Test matrix-vector product"""
    global c, c1
    b0 = b0(N, quad=quad)
    b1 = b1(N, quad=quad)
    mat = inner_product((b0, 0), (b1, k))
    c = mat.matvec(a, c, format='csr')
    c1 = mat.matvec(a, c1, format=format)
    assert np.allclose(c, c1)
    for axis in range(0, dim):
        b, d, d1 = work[dim]
        d.fill(0)
        d1.fill(0)
        d = mat.matvec(b, d, format='csr', axis=axis)
        d1 = mat.matvec(b, d1, format=format, axis=axis)
        assert np.allclose(d, d1)

        # Test multidimensional with axis equals 1D case
        d1.fill(0)
        bc = [np.newaxis,]*dim
        bc[axis] = slice(None)
        fj = np.broadcast_to(a[tuple(bc)], (N,)*dim).copy()
        d1 = mat.matvec(fj, d1, format=format, axis=axis)
        cc = [0,]*dim
        cc[axis] = slice(None)
        assert np.allclose(c, d1[tuple(cc)])

@pytest.mark.parametrize('b0,b1', lbases2)
@pytest.mark.parametrize('quad', ('LG',))
@pytest.mark.parametrize('format', formats)
@pytest.mark.parametrize('dim', (2, 3))
@pytest.mark.parametrize('k0,k1', product((0, 1, 2), (0, 1, 2)))
def test_lmatvec(b0, b1, quad, format, dim, k0, k1):
    """Test matrix-vector product"""
    global c, c1
    b0 = b0(N, quad=quad)
    b1 = b1(N, quad=quad)
    mat = inner_product((b0, k0), (b1, k1))
    c = mat.matvec(a, c, format='csr')
    c1 = mat.matvec(a, c1, format=format)
    assert np.allclose(c, c1)

    for axis in range(0, dim):
        b, d, d1 = work[dim]
        d.fill(0)
        d1.fill(0)
        d = mat.matvec(b, d, format='csr', axis=axis)
        d1 = mat.matvec(b, d1, format=format, axis=axis)
        assert np.allclose(d, d1)

        # Test multidimensional with axis equals 1D case
        d1.fill(0)
        bc = [np.newaxis,]*dim
        bc[axis] = slice(None)
        fj = np.broadcast_to(a[tuple(bc)], (N,)*dim).copy()
        d1 = mat.matvec(fj, d1, format=format, axis=axis)
        cc = [0,]*dim
        cc[axis] = slice(None)
        assert np.allclose(c, d1[tuple(cc)])

@pytest.mark.parametrize('b0,b1', lagbases2)
@pytest.mark.parametrize('quad', lagquads)
@pytest.mark.parametrize('format', ('dia', 'python'))
@pytest.mark.parametrize('dim', (2, 3))
@pytest.mark.parametrize('k0,k1', product((0, 1, 2), (0, 1, 2)))
def test_lagmatvec(b0, b1, quad, format, dim, k0, k1):
    """Test matrix-vector product"""
    global c, c1
    b0 = b0(N, quad=quad)
    b1 = b1(N, quad=quad)
    mat = inner_product((b0, k0), (b1, k1))
    c = mat.matvec(a, c, format='csr')
    c1 = mat.matvec(a, c1, format=format)
    assert np.allclose(c, c1)
    for axis in range(0, dim):
        b, d, d1 = work[dim]
        d.fill(0)
        d1.fill(0)
        d = mat.matvec(b, d, format='csr', axis=axis)
        d1 = mat.matvec(b, d1, format=format, axis=axis)
        assert np.allclose(d, d1)

        # Test multidimensional with axis equals 1D case
        d1.fill(0)
        bc = [np.newaxis,]*dim
        bc[axis] = slice(None)
        fj = np.broadcast_to(a[tuple(bc)], (N,)*dim).copy()
        d1 = mat.matvec(fj, d1, format=format, axis=axis)
        cc = [0,]*dim
        cc[axis] = slice(None)
        assert np.allclose(c, d1[tuple(cc)])

@pytest.mark.parametrize('b0,b1', hbases2)
@pytest.mark.parametrize('quad', hquads)
@pytest.mark.parametrize('format', ('dia', 'python'))
@pytest.mark.parametrize('dim', (2, 3))
@pytest.mark.parametrize('k0,k1', product((0, 1, 2), (0, 1, 2)))
def test_hmatvec(b0, b1, quad, format, dim, k0, k1):
    """Test matrix-vector product"""
    global c, c1
    b0 = b0(N, quad=quad)
    b1 = b1(N, quad=quad)
    mat = inner_product((b0, k0), (b1, k1))
    c = mat.matvec(a, c, format='csr')
    c1 = mat.matvec(a, c1, format=format)
    assert np.allclose(c, c1)
    for axis in range(0, dim):
        b, d, d1 = work[dim]
        d.fill(0)
        d1.fill(0)
        d = mat.matvec(b, d, format='csr', axis=axis)
        d1 = mat.matvec(b, d1, format=format, axis=axis)
        assert np.allclose(d, d1)

        # Test multidimensional with axis equals 1D case
        d1.fill(0)
        bc = [np.newaxis,]*dim
        bc[axis] = slice(None)
        fj = np.broadcast_to(a[tuple(bc)], (N,)*dim).copy()
        d1 = mat.matvec(fj, d1, format=format, axis=axis)
        cc = [0,]*dim
        cc[axis] = slice(None)
        assert np.allclose(c, d1[tuple(cc)])

@pytest.mark.parametrize('b0,b1', jbases2)
@pytest.mark.parametrize('quad', jquads)
@pytest.mark.parametrize('format', ('dia', 'python'))
@pytest.mark.parametrize('dim', (2,))
@pytest.mark.parametrize('k0,k1', product((0, 1, 2), (0, 1, 2)))
def test_jmatvec(b0, b1, quad, format, dim, k0, k1):
    """Testq matrix-vector product"""
    global c, c1
    b0 = b0(N, quad=quad)
    b1 = b1(N, quad=quad)
    mat = inner_product((b0, k0), (b1, k1))
    c = mat.matvec(a, c, format='csr')
    c1 = mat.matvec(a, c1, format=format)
    assert np.allclose(c, c1)
    for axis in range(0, dim):
        b, d, d1 = work[dim]
        d.fill(0)
        d1.fill(0)
        d = mat.matvec(b, d, format='csr', axis=axis)
        d1 = mat.matvec(b, d1, format=format, axis=axis)
        assert np.allclose(d, d1)

        # Test multidimensional with axis equals 1D case
        d1.fill(0)
        bc = [np.newaxis,]*dim
        bc[axis] = slice(None)
        fj = np.broadcast_to(a[tuple(bc)], (N,)*dim).copy()
        d1 = mat.matvec(fj, d1, format=format, axis=axis)
        cc = [0,]*dim
        cc[axis] = slice(None)
        assert np.allclose(c, d1[tuple(cc)])

def test_eq():
    m0 = SparseMatrix({0: 1, 2: 2}, (6, 6))
    m1 = SparseMatrix({0: 1., 2: 2.}, (6, 6))
    assert m0 == m1
    assert m0 is not m1
    m2 = SparseMatrix({0: 1., 2: 3.}, (6, 6))
    assert m0 != m2

@pytest.mark.parametrize('key, mat, quad', mats_and_quads)
def test_imul(key, mat, quad):
    test = key[0]
    trial = key[1]
    measure = 1
    if len(key) == 4:
        domain = key[2]
        measure = key[3]
        if quad == 'GL':
            return
    t0 = test[0]
    t1 = trial[0]
    if len(key) == 4:
        t0 = functools.partial(t0, domain=domain)
        t1 = functools.partial(t1, domain=domain)
    testfunction = (t0(N, quad=quad), test[1])
    trialfunction = (t1(N, quad=quad), trial[1])
    mat = mat(testfunction, trialfunction, measure=measure)
    mc = mat.scale
    mat *= 2
    assert mat.scale == 2.0*mc

    mat.scale = mc
    mat = SparseMatrix(copy(dict(mat)), mat.shape)
    mat *= 2
    assert mat.scale == 2.0

@pytest.mark.parametrize('key, mat, quad', mats_and_quads)
def test_mul(key, mat, quad):
    test = key[0]
    trial = key[1]
    measure = 1
    if len(key) == 4:
        domain = key[2]
        measure = key[3]
        if quad == 'GL':
            return
    t0 = test[0]
    t1 = trial[0]
    if len(key) == 4:
        t0 = functools.partial(t0, domain=domain)
        t1 = functools.partial(t1, domain=domain)
    testfunction = (t0(N, quad=quad), test[1])
    trialfunction = (t1(N, quad=quad), trial[1])
    m = mat(testfunction, trialfunction, measure=measure)
    mc = 2.*m
    assert mc.scale == 2.0*m.scale

    mat = SparseMatrix(copy(dict(m)), m.shape)
    mc = 2.*mat
    assert mc.scale == 2.0

def test_mul2():
    mat = SparseMatrix({0: 1}, (3, 3))
    v = np.ones(3)
    c = mat * v
    assert np.allclose(c, 1)
    mat = SparseMatrix({-2:1, -1:1, 0: 1, 1:1, 2:1}, (3, 3))
    c = mat * v
    assert np.allclose(c, 3)
    SD = FunctionSpace(8, "L", bc=(0, 0), scaled=True)
    u = shenfun.TrialFunction(SD)
    v = shenfun.TestFunction(SD)
    mat = inner(grad(u), grad(v))
    z = Function(SD, val=1)
    c = mat * z
    assert np.allclose(c[:6], 1)

@pytest.mark.parametrize('key, mat, quad', mats_and_quads)
def test_rmul(key, mat, quad):
    test = key[0]
    trial = key[1]
    measure = 1
    if len(key) == 4:
        domain = key[2]
        measure = key[3]
        if quad == 'GL':
            return
    t0 = test[0]
    t1 = trial[0]
    if len(key) == 4:
        t0 = functools.partial(t0, domain=domain)
        t1 = functools.partial(t1, domain=domain)
    testfunction = (t0(N, quad=quad), test[1])
    trialfunction = (t1(N, quad=quad), trial[1])
    m = mat(testfunction, trialfunction, measure=measure)
    mc = m*2.
    assert mc.scale == 2.0*m.scale

    mat = SparseMatrix(copy(dict(m)), m.shape)
    mc = mat*2.
    assert mc.scale == 2.0

@pytest.mark.parametrize('key, mat, quad', mats_and_quads)
def test_div(key, mat, quad):
    test = key[0]
    trial = key[1]
    measure = 1
    if len(key) == 4:
        domain = key[2]
        measure = key[3]
        if quad == 'GL':
            return
    t0 = test[0]
    t1 = trial[0]
    if len(key) == 4:
        t0 = functools.partial(t0, domain=domain)
        t1 = functools.partial(t1, domain=domain)
    testfunction = (t0(N, quad=quad), test[1])
    trialfunction = (t1(N, quad=quad), trial[1])
    m = mat(testfunction, trialfunction, measure=measure)

    mc = m/2.
    assert mc.scale == 0.5*m.scale

    mat = SparseMatrix(copy(dict(m)), m.shape)
    mc = mat/2.
    assert mc.scale == 0.5

@pytest.mark.parametrize('basis, quad', list(product(cBasis, cquads))+
                         list(product(lBasis, lquads))+list(product(lagBasis, lagquads)))
def test_div2(basis, quad):
    B = basis(8, quad=quad)
    u = shenfun.TrialFunction(B)
    v = shenfun.TestFunction(B)
    m = inner(u, v)
    z = Function(B, val=1)
    c = m / z
    m2 = m.diags('csr')
    c2 = spsolve(m2, z[B.slice()])
    assert np.allclose(c2, c[B.slice()])

@pytest.mark.parametrize('key, mat, quad', mats_and_quads)
def test_add(key, mat, quad):
    test = key[0]
    trial = key[1]
    measure = 1
    if len(key) == 4:
        domain = key[2]
        measure = key[3]
        if quad == 'GL':
            return
    t0 = test[0]
    t1 = trial[0]
    if len(key) == 4:
        t0 = functools.partial(t0, domain=domain)
        t1 = functools.partial(t1, domain=domain)
    testfunction = (t0(N, quad=quad), test[1])
    trialfunction = (t1(N, quad=quad), trial[1])
    m = mat(testfunction, trialfunction, measure=measure)

    mc = m + m
    assert mc.scale == 2.0*m.scale

    mat = SparseMatrix(copy(dict(m)), m.shape, m.scale)
    mc = m + mat
    for key, val in mc.items():
        assert np.allclose(val*mc.scale, m[key]*2*m.scale)

@pytest.mark.parametrize('key, mat, quad', mats_and_quads)
def test_iadd(key, mat, quad):
    test = key[0]
    trial = key[1]
    measure = 1
    if len(key) == 4:
        domain = key[2]
        measure = key[3]
        if quad == 'GL':
            return
    t0 = test[0]
    t1 = trial[0]
    if len(key) == 4:
        t0 = functools.partial(t0, domain=domain)
        t1 = functools.partial(t1, domain=domain)
    testfunction = (t0(N, quad=quad), test[1])
    trialfunction = (t1(N, quad=quad), trial[1])
    m = mat(testfunction, trialfunction, measure=measure)
    mc = SparseMatrix(deepcopy(dict(m)), m.shape, m.scale)
    m += mc
    assert np.linalg.norm((m-2*mc).diags('csr').data) < 1e-8

    m -= 2*mc

    assert np.linalg.norm(m.diags('csr').data) < 1e-10

@pytest.mark.parametrize('key, mat, quad', mats_and_quads)
def test_isub(key, mat, quad):
    test = key[0]
    trial = key[1]
    measure = 1
    if len(key) == 4:
        domain = key[2]
        measure = key[3]
        if quad == 'GL':
            return
    t0 = test[0]
    t1 = trial[0]
    if len(key) == 4:
        t0 = functools.partial(t0, domain=domain)
        t1 = functools.partial(t1, domain=domain)
    testfunction = (t0(N, quad=quad), test[1])
    trialfunction = (t1(N, quad=quad), trial[1])
    m = mat(testfunction, trialfunction, measure=measure)
    mc = SparseMatrix(deepcopy(dict(m)), m.shape, m.scale)
    m -= mc
    assert np.linalg.norm(m.diags('csr').data) < 1e-10

    m1 = SparseMatrix(deepcopy(dict(m)), m.shape)
    m2 = SparseMatrix(deepcopy(dict(m)), m.shape)
    m1 -= m2
    assert m1.scale == 0.0

@pytest.mark.parametrize('key, mat, quad', mats_and_quads)
def test_sub(key, mat, quad):
    test = key[0]
    trial = key[1]
    measure = 1
    if len(key) == 4:
        domain = key[2]
        measure = key[3]
        if quad == 'GL':
            return
    t0 = test[0]
    t1 = trial[0]
    if len(key) == 4:
        t0 = functools.partial(t0, domain=domain)
        t1 = functools.partial(t1, domain=domain)
    testfunction = (t0(N, quad=quad), test[1])
    trialfunction = (t1(N, quad=quad), trial[1])
    m = mat(testfunction, trialfunction, measure=measure)
    mc = m - m
    assert mc.scale == 0.0

    m1 = SparseMatrix(copy(dict(m)), m.shape)
    m2 = SparseMatrix(copy(dict(m)), m.shape)

    mc = m1 - m2
    assert mc.scale == 0.0

allaxes2D = {0: (0, 1), 1: (1, 0)}
allaxes3D = {0: (0, 1, 2), 1: (1, 0, 2), 2: (2, 0, 1)}

@pytest.mark.parametrize('axis', (0, 1, 2))
@pytest.mark.parametrize('family', ('chebyshev', 'legendre', 'jacobi'))
def test_helmholtz3D(family, axis):
    la = lla
    if family == 'chebyshev':
        la = cla
    N = (8, 9, 10)
    SD = FunctionSpace(N[allaxes3D[axis][0]], family=family, bc=(0, 0))
    K1 = FunctionSpace(N[allaxes3D[axis][1]], family='F', dtype='D')
    K2 = FunctionSpace(N[allaxes3D[axis][2]], family='F', dtype='d')
    subcomms = mpi4py_fft.pencil.Subcomm(MPI.COMM_WORLD, [0, 1, 1])
    bases = [0]*3
    bases[allaxes3D[axis][0]] = SD
    bases[allaxes3D[axis][1]] = K1
    bases[allaxes3D[axis][2]] = K2
    T = TensorProductSpace(subcomms, bases, axes=allaxes3D[axis])
    u = shenfun.TrialFunction(T)
    v = shenfun.TestFunction(T)
    if family == 'chebyshev':
        mat = inner(v, div(grad(u)))
    else:
        mat = inner(grad(v), grad(u))

    H = la.Helmholtz(*mat)
    u = Function(T)
    s = SD.sl[SD.slice()]
    u[s] = np.random.random(u[s].shape) + 1j*np.random.random(u[s].shape)
    f = Function(T)
    f = H.matvec(u, f)

    g0 = Function(T)
    g1 = Function(T)
    M = {d.get_key(): d for d in mat}
    g0 = M['ADDmat'].matvec(u, g0)
    g1 = M['BDDmat'].matvec(u, g1)

    assert np.linalg.norm(f-(g0+g1)) < 1e-12, np.linalg.norm(f-(g0+g1))

    uc = Function(T)
    uc = H(uc, f)
    assert np.linalg.norm(uc-u) < 1e-12

@pytest.mark.parametrize('axis', (0, 1))
@pytest.mark.parametrize('family', ('chebyshev', 'legendre', 'jacobi'))
def test_helmholtz2D(family, axis):
    la = lla
    if family == 'chebyshev':
        la = cla
    N = (8, 9)
    SD = FunctionSpace(N[axis], family=family, bc=(0, 0))
    K1 = FunctionSpace(N[(axis+1)%2], family='F', dtype='d')
    subcomms = mpi4py_fft.pencil.Subcomm(MPI.COMM_WORLD, allaxes2D[axis])
    bases = [K1]
    bases.insert(axis, SD)
    T = TensorProductSpace(subcomms, bases, axes=allaxes2D[axis])
    u = shenfun.TrialFunction(T)
    v = shenfun.TestFunction(T)
    if family == 'chebyshev':
        mat = inner(v, div(grad(u)))
    else:
        mat = inner(grad(v), grad(u))

    H = la.Helmholtz(*mat)
    u = Function(T)
    s = SD.sl[SD.slice()]
    u[s] = np.random.random(u[s].shape) + 1j*np.random.random(u[s].shape)
    f = Function(T)
    f = H.matvec(u, f)

    g0 = Function(T)
    g1 = Function(T)
    M = {d.get_key(): d for d in mat}
    g0 = M['ADDmat'].matvec(u, g0)
    g1 = M['BDDmat'].matvec(u, g1)

    assert np.linalg.norm(f-(g0+g1)) < 1e-12, np.linalg.norm(f-(g0+g1))

    uc = Function(T)
    uc = H(uc, f)
    assert np.linalg.norm(uc-u) < 1e-12

@pytest.mark.parametrize('axis', (0, 1, 2))
@pytest.mark.parametrize('family', ('chebyshev', 'legendre', 'jacobi'))
def test_biharmonic3D(family, axis):
    la = lla
    if family == 'chebyshev':
        la = cla
    N = (16, 16, 16)
    SD = FunctionSpace(N[allaxes3D[axis][0]], family=family, bc='Biharmonic')
    K1 = FunctionSpace(N[allaxes3D[axis][1]], family='F', dtype='D')
    K2 = FunctionSpace(N[allaxes3D[axis][2]], family='F', dtype='d')
    subcomms = mpi4py_fft.pencil.Subcomm(MPI.COMM_WORLD, [0, 1, 1])
    bases = [0]*3
    bases[allaxes3D[axis][0]] = SD
    bases[allaxes3D[axis][1]] = K1
    bases[allaxes3D[axis][2]] = K2
    T = TensorProductSpace(subcomms, bases, axes=allaxes3D[axis])
    u = shenfun.TrialFunction(T)
    v = shenfun.TestFunction(T)
    if family == 'chebyshev':
        mat = inner(v, div(grad(div(grad(u)))))
    else:
        mat = inner(div(grad(v)), div(grad(u)))

    H = la.Biharmonic(*mat)
    u = Function(T)
    u[:] = np.random.random(u.shape) + 1j*np.random.random(u.shape)
    f = Function(T)
    f = H.matvec(u, f)

    g0 = Function(T)
    g1 = Function(T)
    g2 = Function(T)
    M = {d.get_key(): d for d in mat}
    g0 = M['SBBmat'].matvec(u, g0)
    g1 = M['ABBmat'].matvec(u, g1)
    g2 = M['BBBmat'].matvec(u, g2)

    assert np.linalg.norm(f-(g0+g1+g2)) < 1e-8, np.linalg.norm(f-(g0+g1+g2))

@pytest.mark.parametrize('axis', (0, 1))
@pytest.mark.parametrize('family', ('chebyshev', 'legendre', 'jacobi'))
def test_biharmonic2D(family, axis):
    la = lla
    if family == 'chebyshev':
        la = cla
    N = (16, 16)
    SD = FunctionSpace(N[axis], family=family, bc='Biharmonic')
    K1 = FunctionSpace(N[(axis+1)%2], family='F', dtype='d')
    subcomms = mpi4py_fft.pencil.Subcomm(MPI.COMM_WORLD, allaxes2D[axis])
    bases = [K1]
    bases.insert(axis, SD)
    T = TensorProductSpace(subcomms, bases, axes=allaxes2D[axis])
    u = shenfun.TrialFunction(T)
    v = shenfun.TestFunction(T)
    if family == 'chebyshev':
        mat = inner(v, div(grad(div(grad(u)))))
    else:
        mat = inner(div(grad(v)), div(grad(u)))

    H = la.Biharmonic(*mat)
    u = Function(T)
    u[:] = np.random.random(u.shape) + 1j*np.random.random(u.shape)
    f = Function(T)
    f = H.matvec(u, f)

    g0 = Function(T)
    g1 = Function(T)
    g2 = Function(T)
    M = {d.get_key(): d for d in mat}
    g0 = M['SBBmat'].matvec(u, g0)
    g1 = M['ABBmat'].matvec(u, g1)
    g2 = M['BBBmat'].matvec(u, g2)

    assert np.linalg.norm(f-(g0+g1+g2)) < 1e-8


if __name__ == '__main__':
    import sympy as sp
    x = sp.symbols('x', real=True, positive=True)
    test_mat(((cbases.ShenDirichletBasis, 0), (cbases.BCBasis, 0)), cmatrices.BCDmat, 'GC')
    #test_cmatvec(cBasis[3], cBasis[1], 'GC', 'cython', 3, 0)
    #test_lagmatvec(lagBasis[0], lagBasis[1], 'LG', 'python', 3, 2, 0)
    #test_hmatvec(hBasis[0], hBasis[0], 'HG', 'self', 3, 1, 1)
    #test_iadd(*mats_and_quads[15])
    #test_sub(*mats_and_quads[15])
    #test_mul2()
    #test_div2(cBasis[0], 'GC')
    #test_helmholtz2D('chebyshev', 0)
    #test_helmholtz3D('chebyshev', 0)
    #test_biharmonic3D('chebyshev', 0)
    #test_biharmonic2D('jacobi', 0)
