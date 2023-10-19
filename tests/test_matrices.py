from copy import copy, deepcopy
import functools
from ipaddress import ip_address
from itertools import product
import numpy as np
import sympy as sp
from mpi4py import MPI
import pytest
import mpi4py_fft
import shenfun
from shenfun.chebyshev import matrices as cmatrices
from shenfun.chebyshev import bases as cbases
from shenfun.chebyshevu import matrices as cumatrices
from shenfun.chebyshevu import bases as cubases
from shenfun.ultraspherical import matrices as umatrices
from shenfun.ultraspherical import bases as ubases
from shenfun.legendre import matrices as lmatrices
from shenfun.legendre import bases as lbases
from shenfun.laguerre import matrices as lagmatrices
from shenfun.laguerre import bases as lagbases
from shenfun.hermite import matrices as hmatrices
from shenfun.hermite import bases as hbases
from shenfun.jacobi import matrices as jmatrices
from shenfun.jacobi import bases as jbases
from shenfun.chebyshev import la as cla
from shenfun import div, grad, inner, TensorProductSpace, FunctionSpace, SparseMatrix,\
    Function, comm, VectorSpace, BlockMatrix, CompositeSpace
from shenfun.spectralbase import inner_product
from shenfun.config import config

x = sp.symbols('x', real=True)
xp = sp.symbols('x', real=True, positive=True)

ctrialBasis = [cbases.__dict__.get(base) for base in cbases.bases[:-1]]
ctestBasis = ctrialBasis + [cbases.__dict__.get(base) for base in cbases.testbases[:-1]]
cutrialBasis = [cubases.__dict__.get(base) for base in cubases.bases[:-1]]
cutestBasis = cutrialBasis + [cubases.__dict__.get(base) for base in cubases.testbases[:-1]]

utrialBasis = [ubases.__dict__.get(base) for base in ubases.bases[:-1]]
utestBasis = utrialBasis + [ubases.__dict__.get(base) for base in ubases.testbases[:-1]]
ltrialBasis = [lbases.__dict__.get(base) for base in lbases.bases[:-1]]
ltestBasis = ltrialBasis + [lbases.__dict__.get(base) for base in lbases.testbases[:-1]]
latrialBasis = [lagbases.__dict__.get(base) for base in lagbases.bases[:-1]]
htrialBasis = [hbases.__dict__.get(base) for base in hbases.bases]
jtrialBasis = [jbases.__dict__.get(base) for base in jbases.bases[:-1]]
jtestBasis = jtrialBasis + [jbases.__dict__.get(base) for base in jbases.testbases]

bcbases = (
    cbases.BCGeneric,
    lbases.BCGeneric,
    cubases.BCGeneric,
    ubases.BCGeneric
)

cquads = ('GC', 'GL')
cuquads = ('GU', )
uquads = ('QG',)
lquads = ('LG', 'GL')
lagquads = ('LG',)
hquads = ('HG',)
jquads = ('JG',)

bcs = {
    0: {'left': {'D': 1}, 'right': {'D': 1}},
    1: {'left': {'N': 1}, 'right': {'N': 1}},
    2: {'left': {'D': 1}, 'right': {'N': 1}},
    3: {'left': {'N': 1}, 'right': {'D': 1}},
    4: {'left': {'D': 1, 'N': 1}},
    5: {'right': {'D': 1, 'N': 1}}
}

for f in ['dct', 'dst', 'fft', 'ifft', 'rfft', 'irfft']:
    config['fftw'][f]['planner_effort'] = 'FFTW_ESTIMATE'

N = 14
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

cbases2 = list(product(ctestBasis, ctrialBasis))
cubases2 = list(product(cutestBasis, cutrialBasis))
ubases2 = list(product(utestBasis, utrialBasis))
lbases2 = list(product(ltestBasis, ltrialBasis))
lagbases2 = list(product(latrialBasis, latrialBasis))
hbases2 = list(product(htrialBasis, htrialBasis))
jbases2 = list(product(jtestBasis, jtrialBasis))
bases2 = cbases2+lbases2+lagbases2+hbases2+jbases2+cubases2+ubases2

cmats_and_quads = [list(k[0])+[k[1]] for k in product([(k, v) for k, v in cmatrices.mat.items()], cquads)]
cumats_and_quads = [list(k[0])+[k[1]] for k in product([(k, v) for k, v in cumatrices.mat.items()], cuquads)]
umats_and_quads = [list(k[0])+[k[1]] for k in product([(k, v) for k, v in umatrices.mat.items()], uquads)]
lmats_and_quads = [list(k[0])+[k[1]] for k in product([(k, v) for k, v in lmatrices.mat.items()], lquads)]
lagmats_and_quads = [list(k[0])+[k[1]] for k in product([(k, v) for k, v in lagmatrices.mat.items()], lagquads)]
hmats_and_quads = [list(k[0])+[k[1]] for k in product([(k, v) for k, v in hmatrices.mat.items()], hquads)]
jmats_and_quads = [list(k[0])+[k[1]] for k in product([(k, v) for k, v in jmatrices.mat.items()], jquads)]
mats_and_quads = cmats_and_quads+lmats_and_quads+lagmats_and_quads+hmats_and_quads+jmats_and_quads+umats_and_quads+cumats_and_quads

#cmats_and_quads_ids = ['-'.join(i) for i in product([v.__name__ for v in cmatrices.mat.values()], cquads)]
#lmats_and_quads_ids = ['-'.join(i) for i in product([v.__name__ for v in lmatrices.mat.values()], lquads)]

num_tests = 10
some_mats_and_quads = [mats_and_quads[i] for i in np.random.randint(0, len(mats_and_quads), num_tests)]
some_cbases2 = [cbases2[i] for i in np.random.randint(0, len(cbases2), num_tests)]
some_cubases2 = [cubases2[i] for i in np.random.randint(0, len(cubases2), num_tests)]
some_lbases2 = [lbases2[i] for i in np.random.randint(0, len(lbases2), num_tests)]
some_jbases2 = [jbases2[i] for i in np.random.randint(0, len(jbases2), num_tests)]


@pytest.mark.parametrize('key, mat, quad', mats_and_quads)
def test_mat(key, mat, quad):
    """Test that matrices built by hand equals those automatically generated"""
    test = key[0]
    trial = key[1]
    measure = 1
    if len(key) == 3:
        measure = key[2]

    t0 = test[0]
    t1 = trial[0]

    if trial[0] in bcbases:
        # Just use some random boundary condition
        t1 = functools.partial(t1, bc=bcs[np.random.randint(0, 6)])

    testfunction = (t0(N, quad=quad), test[1])
    trialfunction = (t1(N, quad=quad), trial[1])
    mat = mat(testfunction, trialfunction, measure=measure)
    fixed_resolution = None if sp.degree(sp.sympify(measure)) < 2 else int(3*N/2)
    shenfun.check_sanity(mat, testfunction, trialfunction, measure, fixed_resolution=fixed_resolution)
    if test[0].family() == 'Legendre' and test[0].boundary_condition() == 'Dirichlet':
        testfunction = (test[0](N, quad=quad, scaled=True), test[1])
        trialfunction = (trial[0](N, quad=quad, scaled=True), trial[1])
        mat = mat(testfunction, trialfunction)
        shenfun.check_sanity(mat, testfunction, trialfunction, measure)


@pytest.mark.parametrize('b0,b1', some_cbases2)
@pytest.mark.parametrize('quad', cquads)
@pytest.mark.parametrize('k', range(5))
def test_cmatvec(b0, b1, quad, k):
    """Test matrix-vector product"""
    global c, c1
    b0 = b0(N, quad=quad)
    b1 = b1(N, quad=quad)
    mat = inner_product((b0, 0), (b1, k))
    formats = mat._matvec_methods + ['python', 'csr']
    c = mat.matvec(a, c, format='csr')
    for format in formats:
        c1 = mat.matvec(a, c1, format=format)
        assert np.allclose(c, c1)
    for dim in (2, 3):
        b, d, d1 = work[dim]
        for axis in range(0, dim):
            d = mat.matvec(b, d, format='csr', axis=axis)
            for format in formats:
                d1 = mat.matvec(b, d1, format=format, axis=axis)
                assert np.allclose(d, d1)

@pytest.mark.parametrize('b0,b1', some_cubases2)
@pytest.mark.parametrize('quad', cuquads)
@pytest.mark.parametrize('k', range(5))
def test_cumatvec(b0, b1, quad, k):
    """Test matrix-vector product"""
    global c, c1
    b0 = b0(N, quad=quad)
    b1 = b1(N, quad=quad)
    mat = inner_product((b0, 0), (b1, k))
    formats = mat._matvec_methods + ['python', 'csr']
    c = mat.matvec(a, c, format='csr')
    for format in formats:
        c1 = mat.matvec(a, c1, format=format)
        assert np.allclose(c, c1)
    for dim in (2, 3):
        b, d, d1 = work[dim]
        for axis in range(0, dim):
            d = mat.matvec(b, d, format='csr', axis=axis)
            for format in formats:
                d1 = mat.matvec(b, d1, format=format, axis=axis)
                assert np.allclose(d, d1)

@pytest.mark.parametrize('b0,b1', some_lbases2)
@pytest.mark.parametrize('quad', ('LG',))
@pytest.mark.parametrize('k0,k1', product((0, 1, 2), (0, 1, 2)))
def test_lmatvec(b0, b1, quad, k0, k1):
    """Test matrix-vector product"""
    global c, c1, a
    b0 = b0(N, quad=quad)
    b1 = b1(N, quad=quad)
    mat = inner_product((b0, k0), (b1, k1))
    c = mat.matvec(a, c, format='dia')
    formats = mat._matvec_methods + ['python', 'csr']
    for format in formats:
        c1 = mat.matvec(a, c1, format=format)
        assert np.allclose(c, c1)

    for dim in (2, 3):
        b, d, d1 = work[dim]
        for axis in range(0, dim):
            d = mat.matvec(b, d, format='dia', axis=axis)
            for formats in formats:
                d1 = mat.matvec(b, d1, format=format, axis=axis)
                assert np.allclose(d, d1)


@pytest.mark.parametrize('b0,b1', lagbases2)
@pytest.mark.parametrize('quad', lagquads)
@pytest.mark.parametrize('format', ('dia', 'python'))
@pytest.mark.parametrize('k0,k1', product((0, 1, 2), (0, 1, 2)))
def test_lagmatvec(b0, b1, quad, format, k0, k1):
    """Test matrix-vector product"""
    global c, c1
    b0 = b0(N, quad=quad)
    b1 = b1(N, quad=quad)
    mat = inner_product((b0, k0), (b1, k1))
    c = mat.matvec(a, c, format='csr')
    formats = mat._matvec_methods + ['python', 'csr']
    for format in formats:
        c1 = mat.matvec(a, c1, format=format)
        assert np.allclose(c, c1)
    for dim in (2, 3):
        b, d, d1 = work[dim]
        for axis in range(0, dim):
            d = mat.matvec(b, d, format='csr', axis=axis)
            for format in formats:
                d1 = mat.matvec(b, d1, format=format, axis=axis)
                assert np.allclose(d, d1)

@pytest.mark.parametrize('b0,b1', hbases2)
@pytest.mark.parametrize('quad', hquads)
@pytest.mark.parametrize('format', ('dia', 'python'))
@pytest.mark.parametrize('k0,k1', product((0, 1, 2), (0, 1, 2)))
def test_hmatvec(b0, b1, quad, format, k0, k1):
    """Test matrix-vector product"""
    global c, c1
    b0 = b0(N, quad=quad)
    b1 = b1(N, quad=quad)
    mat = inner_product((b0, k0), (b1, k1))
    formats = mat._matvec_methods + ['python', 'csr']
    c = mat.matvec(a, c, format='csr')
    for format in formats:
        c1 = mat.matvec(a, c1, format=format)
        assert np.allclose(c, c1)
    for dim in (2, 3):
        b, d, d1 = work[dim]
        for axis in range(0, dim):
            d = mat.matvec(b, d, format='csr', axis=axis)
            for format in formats:
                d1 = mat.matvec(b, d1, format=format, axis=axis)
                assert np.allclose(d, d1)

@pytest.mark.parametrize('b0,b1', some_jbases2)
@pytest.mark.parametrize('quad', jquads)
@pytest.mark.parametrize('format', ('dia', 'python'))
@pytest.mark.parametrize('k0,k1', product((0, 1, 2), (0, 1, 2)))
def test_jmatvec(b0, b1, quad, format, k0, k1):
    """Test matrix-vector product"""
    global c, c1
    b0 = b0(N, quad=quad)
    b1 = b1(N, quad=quad)

    mat = inner_product((b0, k0), (b1, k1))
    c = mat.matvec(a, c, format='csr')
    formats = mat._matvec_methods + ['python', 'csr']
    for format in formats:
        c1 = mat.matvec(a, c1, format=format)
        assert np.allclose(c, c1)
    dim = 2
    b, d, d1 = work[dim]
    for axis in range(0, dim):
        d = mat.matvec(b, d, format='csr', axis=axis)
        for format in formats:
            d1 = mat.matvec(b, d1, format=format, axis=axis)
            assert np.allclose(d, d1)

def test_eq():
    m0 = SparseMatrix({0: 1, 2: 2}, (6, 6))
    m1 = SparseMatrix({0: 1., 2: 2.}, (6, 6))
    assert m0 == m1
    assert m0 is not m1
    m2 = SparseMatrix({0: 1., 2: 3.}, (6, 6))
    assert m0 != m2

@pytest.mark.parametrize('key, mat, quad', some_mats_and_quads)
def test_imul(key, mat, quad):
    test = key[0]
    trial = key[1]
    measure = 1
    if len(key) == 3:
        measure = key[2]
        if quad == 'GL':
            return

    t0 = test[0]
    t1 = trial[0]
    if trial[0] in bcbases:
        # Just use some random boundary condition
        t1 = functools.partial(t1, bc=bcs[np.random.randint(0, 6)])
    testfunction = (t0(N, quad=quad), test[1])
    trialfunction = (t1(N, quad=quad), trial[1])
    try:
        mat = mat(testfunction, trialfunction, measure=measure)
    except AssertionError:
        return
    mc = mat.scale
    mat *= 2
    assert mat.scale == 2.0*mc

    mat.scale = mc
    mat = SparseMatrix(copy(dict(mat)), mat.shape)
    mat *= 2
    assert mat.scale == 2.0

@pytest.mark.parametrize('key, mat, quad', some_mats_and_quads)
def test_mul(key, mat, quad):
    test = key[0]
    trial = key[1]
    measure = 1
    if len(key) == 3:
        measure = key[2]
        if quad == 'GL':
            return

    t0 = test[0]
    t1 = trial[0]
    if trial[0] in bcbases:
        # Just use some random boundary condition
        t1 = functools.partial(t1, bc=bcs[np.random.randint(0, 6)])
    testfunction = (t0(N, quad=quad), test[1])
    trialfunction = (t1(N, quad=quad), trial[1])
    try:
        m = mat(testfunction, trialfunction, measure=measure)
    except AssertionError:
        return
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

@pytest.mark.parametrize('key, mat, quad', some_mats_and_quads)
def test_rmul(key, mat, quad):
    test = key[0]
    trial = key[1]
    measure = 1
    if len(key) == 3:
        measure = key[2]
        if quad == 'GL':
            return

    t0 = test[0]
    t1 = trial[0]
    if trial[0] in bcbases:
        # Just use some random boundary condition
        t1 = functools.partial(t1, bc=bcs[np.random.randint(0, 6)])
    testfunction = (t0(N, quad=quad), test[1])
    trialfunction = (t1(N, quad=quad), trial[1])
    try:
        m = mat(testfunction, trialfunction, measure=measure)
    except AssertionError:
        return
    mc = m*2.
    assert mc.scale == 2.0*m.scale

    mat = SparseMatrix(copy(dict(m)), m.shape)
    mc = mat*2.
    assert mc.scale == 2.0

@pytest.mark.parametrize('key, mat, quad', some_mats_and_quads)
def test_div(key, mat, quad):
    test = key[0]
    trial = key[1]
    measure = 1
    if len(key) == 3:
        measure = key[2]
        if quad == 'GL':
            return

    t0 = test[0]
    t1 = trial[0]
    if trial[0] in bcbases:
        # Just use some random boundary condition
        t1 = functools.partial(t1, bc=bcs[np.random.randint(0, 6)])
    testfunction = (t0(N, quad=quad), test[1])
    trialfunction = (t1(N, quad=quad), trial[1])
    try:
        m = mat(testfunction, trialfunction, measure=measure)
    except AssertionError:
        return

    mc = m/2.
    assert mc.scale == 0.5*m.scale

    mat = SparseMatrix(copy(dict(m)), m.shape)
    mc = mat/2.
    assert mc.scale == 0.5

@pytest.mark.parametrize('basis, quad', list(product(ctrialBasis, cquads))+
                         list(product(ltrialBasis, lquads))+list(product(latrialBasis, lagquads)))
def test_div2(basis, quad):
    B = basis(10, quad=quad)
    u = shenfun.TrialFunction(B)
    v = shenfun.TestFunction(B)
    m = inner(u, v)
    z = Function(B, val=1)
    c = m / z
    #m2 = m.diags('csr')
    #c2 = spsolve(m2, z[B.slice()])

    c2 = Function(B)
    c2 = m.solve(z, c2)
    assert np.allclose(c2[B.slice()], c[B.slice()])

@pytest.mark.parametrize('key, mat, quad', some_mats_and_quads)
def test_add(key, mat, quad):
    test = key[0]
    trial = key[1]
    measure = 1
    if len(key) == 3:
        measure = key[2]
        if quad == 'GL':
            return

    t0 = test[0]
    t1 = trial[0]
    if trial[0] in bcbases:
        # Just use some random boundary condition
        t1 = functools.partial(t1, bc=bcs[np.random.randint(0, 6)])
    testfunction = (t0(N, quad=quad), test[1])
    trialfunction = (t1(N, quad=quad), trial[1])
    try:
        m = mat(testfunction, trialfunction, measure=measure)
    except AssertionError:
        return

    mc = m + m
    assert np.linalg.norm(mc.diags('csr').data-m.diags('csr').data*2) < 1e-8

    mat = SparseMatrix(copy(dict(m)), m.shape, m.scale)
    mc = m + mat
    assert np.linalg.norm(mc.diags('csr').data-m.diags('csr').data*2) < 1e-8

@pytest.mark.parametrize('key, mat, quad', some_mats_and_quads)
def test_iadd(key, mat, quad):
    test = key[0]
    trial = key[1]
    measure = 1
    if len(key) == 3:
        measure = key[2]
        if quad == 'GL':
            return

    t0 = test[0]
    t1 = trial[0]
    if trial[0] in bcbases:
        # Just use some random boundary condition
        t1 = functools.partial(t1, bc=bcs[np.random.randint(0, 6)])
    testfunction = (t0(N, quad=quad), test[1])
    trialfunction = (t1(N, quad=quad), trial[1])
    try:
        m = mat(testfunction, trialfunction, measure=measure)
    except AssertionError:
        return
    mc = m.copy()
    m += mc
    assert np.linalg.norm(m.diags('csr').data-mc.diags('csr').data*2) < 1e-8
    m -= 2*mc
    assert np.linalg.norm(m.diags('csr').data) < 1e-8

@pytest.mark.parametrize('key, mat, quad', some_mats_and_quads)
def test_isub(key, mat, quad):
    test = key[0]
    trial = key[1]
    measure = 1
    if len(key) == 3:
        measure = key[2]
        if quad == 'GL':
            return
    t0 = test[0]
    t1 = trial[0]
    if trial[0] in bcbases:
        # Just use some random boundary condition
        t1 = functools.partial(t1, bc=bcs[np.random.randint(0, 6)])
    testfunction = (t0(N, quad=quad), test[1])
    trialfunction = (t1(N, quad=quad), trial[1])
    try:
        m = mat(testfunction, trialfunction, measure=measure)
    except AssertionError:
        return
    mc = m.copy()
    m -= mc
    assert np.linalg.norm(m.diags('csr').data) < 1e-8

    m1 = SparseMatrix(deepcopy(dict(mc)), m.shape)
    m2 = SparseMatrix(deepcopy(dict(mc)), m.shape)
    m1 -= m2
    assert np.linalg.norm(m1.diags('csr').data) < 1e-8

@pytest.mark.parametrize('key, mat, quad', some_mats_and_quads)
def test_sub(key, mat, quad):
    test = key[0]
    trial = key[1]
    measure = 1
    if len(key) == 3:
        measure = key[2]
        if quad == 'GL':
            return
    t0 = test[0]
    t1 = trial[0]
    if trial[0] in bcbases:
        # Just use some random boundary condition
        t1 = functools.partial(t1, bc=bcs[np.random.randint(0, 6)])
    testfunction = (t0(N, quad=quad), test[1])
    trialfunction = (t1(N, quad=quad), trial[1])
    try:
        m = mat(testfunction, trialfunction, measure=measure)
    except AssertionError:
        return
    mc = m - m
    assert np.linalg.norm(mc.diags('csr').data) < 1e-8

    m1 = SparseMatrix(copy(dict(m)), m.shape)
    m2 = SparseMatrix(copy(dict(m)), m.shape)

    mc = m1 - m2
    assert np.linalg.norm(mc.diags('csr').data) < 1e-8

allaxes2D = {0: (0, 1), 1: (1, 0)}
allaxes3D = {0: (0, 1, 2), 1: (1, 0, 2), 2: (2, 0, 1)}

@pytest.mark.parametrize('axis', (0, 1, 2))
@pytest.mark.parametrize('family', ('chebyshev',))
def test_helmholtz3D(family, axis):
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
    T = TensorProductSpace(subcomms, bases, axes=allaxes3D[axis], modify_spaces_inplace=True)
    u = shenfun.TrialFunction(T)
    v = shenfun.TestFunction(T)
    mat = inner(v, div(grad(u)))
    G = shenfun.la.Solver3D(mat)

    H = la.Helmholtz(*mat)
    uh = Function(T)
    s = SD.sl[SD.slice()]
    uh[s] = np.random.random(uh[s].shape) + 1j*np.random.random(uh[s].shape)
    f = Function(T)
    f = H.matvec(uh, f)

    gg = Function(T)
    gg = G.matvec(uh, gg)
    assert np.linalg.norm(f-gg) < 1e-12, np.linalg.norm(f-gg)

    uc = Function(T)
    uc = H(f, uc)
    assert np.linalg.norm(uc-uh) < 1e-12
    T.destroy()

@pytest.mark.parametrize('axis', (0, 1))
@pytest.mark.parametrize('family', ('chebyshev',))
def test_helmholtz2D(family, axis):
    la = cla
    N = (8, 9)
    SD = FunctionSpace(N[axis], family=family, bc=(0, 0))
    K1 = FunctionSpace(N[(axis+1)%2], family='F', dtype='d')
    subcomms = mpi4py_fft.pencil.Subcomm(MPI.COMM_WORLD, allaxes2D[axis])
    bases = [K1]
    bases.insert(axis, SD)
    T = TensorProductSpace(subcomms, bases, axes=allaxes2D[axis], modify_spaces_inplace=True)
    u = shenfun.TrialFunction(T)
    v = shenfun.TestFunction(T)
    mat = inner(v, div(grad(u)))
    H = la.Helmholtz(*mat)
    G = shenfun.la.Solver2D(mat)
    uh = Function(T)
    s = SD.sl[SD.slice()]
    uh[s] = np.random.random(uh[s].shape) + 1j*np.random.random(uh[s].shape)
    f = Function(T)
    f = H.matvec(uh, f)
    gg = Function(T)
    gg = G.matvec(uh, gg)
    assert np.linalg.norm(f-gg) < 1e-12, np.linalg.norm(f-gg)
    uc = Function(T)
    uc = H(f, uc)
    assert np.linalg.norm(uc-uh) < 1e-12
    T.destroy()

@pytest.mark.parametrize('axis', (0, 1, 2))
@pytest.mark.parametrize('family', ('chebyshev',))
def test_biharmonic3D(family, axis):
    la = cla
    N = (16, 16, 16)
    SD = FunctionSpace(N[allaxes3D[axis][0]], family=family, bc=(0, 0, 0, 0))
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
    mat = inner(v, div(grad(div(grad(u)))))

    H = la.Biharmonic(*mat)
    u = Function(T)
    u[:] = np.random.random(u.shape) + 1j*np.random.random(u.shape)
    f = Function(T)
    f = H.matvec(u, f)

    g0 = Function(T)
    g1 = Function(T)
    g2 = Function(T)
    mat = H.tpmats
    M = {d.get_key(): d for d in mat}
    g0 = M['SSBSBmat'].matvec(u, g0)
    g1 = M['ASBSBmat'].matvec(u, g1)
    g2 = M['BSBSBmat'].matvec(u, g2)
    T.destroy()

    assert np.linalg.norm(f-(g0+g1+g2)) < 1e-8, np.linalg.norm(f-(g0+g1+g2))

@pytest.mark.parametrize('axis', (0, 1))
@pytest.mark.parametrize('family', ('chebyshev',))
def test_biharmonic2D(family, axis):
    la = cla
    N = (16, 16)
    SD = FunctionSpace(N[axis], family=family, bc=(0, 0, 0, 0))
    K1 = FunctionSpace(N[(axis+1)%2], family='F', dtype='d')
    subcomms = mpi4py_fft.pencil.Subcomm(MPI.COMM_WORLD, allaxes2D[axis])
    bases = [K1]
    bases.insert(axis, SD)
    T = TensorProductSpace(subcomms, bases, axes=allaxes2D[axis])
    u = shenfun.TrialFunction(T)
    v = shenfun.TestFunction(T)
    mat = inner(v, div(grad(div(grad(u)))))

    H = la.Biharmonic(*mat)
    u = Function(T)
    u[:] = np.random.random(u.shape) + 1j*np.random.random(u.shape)
    f = Function(T)
    f = H.matvec(u, f)

    g0 = Function(T)
    g1 = Function(T)
    g2 = Function(T)
    mat = H.tpmats
    M = {d.get_key(): d for d in mat}
    g0 = M['SSBSBmat'].matvec(u, g0)
    g1 = M['ASBSBmat'].matvec(u, g1)
    g2 = M['BSBSBmat'].matvec(u, g2)
    T.destroy()
    assert np.linalg.norm(f-(g0+g1+g2)) < 1e-8


D = FunctionSpace(8, 'C', bc=(0, 0))
D0 = FunctionSpace(8, 'C')
F = FunctionSpace(8, 'F', dtype='d')
F2 = FunctionSpace(8, 'F', dtype='D')

@pytest.mark.parametrize('bases', ((D, D0), (D, F), (D, D0), (D, D0, F),
                                   (D, F2, F), (D0, F2, F)))
def test_blockmatrix(bases):
    T = TensorProductSpace(comm, bases)
    V = VectorSpace(T)
    u = shenfun.TrialFunction(V)
    v = shenfun.TestFunction(V)
    A = inner(u, v)
    B = BlockMatrix(A)
    uh = Function(V, val=1)
    c = Function(V)
    c = B.matvec(uh, c, use_scipy=True)
    c2 = Function(V)
    c2 = B.matvec(uh, c2, use_scipy=False)
    assert np.linalg.norm(c2-c) < 1e-8
    VQ = CompositeSpace([V, T])
    u, p = shenfun.TrialFunction(VQ)
    v, q = shenfun.TestFunction(VQ)
    A2 = inner(u, v) + [inner(p, q)]
    B2 = BlockMatrix(A2)
    uh = Function(VQ, val=1)
    c = Function(VQ)
    c = B2.matvec(uh, c, use_scipy=True)
    c2 = Function(VQ)
    c2 = B2.matvec(uh, c2, use_scipy=False)
    assert np.linalg.norm(c2-c) < 1e-8
    T.destroy()

@pytest.mark.parametrize('b0,b1', some_cbases2+some_lbases2)
def test_stencil(b0, b1):
    N = 14
    b0 = b0(N)
    b1 = b1(N)
    u = shenfun.TrialFunction(b1)
    v = shenfun.TestFunction(b0)
    B0 = inner(v, u, kind='vandermonde')
    B1 = inner(v, u, kind='stencil')
    C = B0-B1
    C.incorporate_scale()
    assert np.linalg.norm(C.diags('csr').data) < 1e-8
    B0 = inner(v*x, u, kind='vandermonde')
    B1 = inner(v*x, u, kind='stencil')
    C = B0-B1
    C.incorporate_scale()
    assert np.linalg.norm(C.diags('csr').data) < 1e-8

@pytest.mark.parametrize('b0,b1', some_lbases2+some_cbases2)
def test_adaptive_quadrature(b0, b1):
    N = 12
    b0 = b0(N)
    b1 = b1(N)
    u = shenfun.TrialFunction(b1)
    v = shenfun.TestFunction(b0)
    B0 = inner(v, u, assemble='adaptive')
    B1 = inner(v, u, assemble='quadrature', kind='stencil')
    C = B0-B1
    C.incorporate_scale()
    assert np.linalg.norm(C.diags('csr').data) < 1e-8

# Some bases take too long for Chebyshev, so just choose these three
some_cbases3 = [(cbases.ShenDirichlet, cbases.ShenDirichlet),
                (cbases.ShenDirichlet, cbases.ShenNeumann),
                (cbases.LowerDirichlet, cbases.UpperDirichlet)]
some_lbases3 = [lbases2[i] for i in np.random.randint(0, len(lbases2), 3)]

@pytest.mark.parametrize('b0,b1', some_lbases3+some_cbases3)
def test_exact(b0, b1):
    N = 10
    b0 = b0(N)
    b1 = b1(N)
    u = shenfun.TrialFunction(b1)
    v = shenfun.TestFunction(b0)
    B0 = inner(v, u, assemble='exact')
    B1 = inner(v, u, fixed_resolution=20)
    C = B0-B1
    C.incorporate_scale()
    assert np.linalg.norm(C.diags('csr').data) < 1e-8

if __name__ == '__main__':
    import sympy as sp
    x = sp.symbols('x', real=True)
    xp = sp.Symbol('x', real=True, positive=True)

    #test_exact(ltestBasis[-3], ltrialBasis[8])
    #test_mat(((ctestBasis[1], 0), (ctestBasis[1], 2), 1-x**2), cmatrices.ASDSDmatW, 'GC')
    #test_mat(((cbases.Orthogonal, 0), (lbases.Orthogonal, 0)), cmatrices.BTLmat, 'GL')
    #test_cmatvec(ctrialBasis[1], ctrialBasis[1], 'GC', 0)
    #test_lmatvec(ltrialBasis[0], ltrialBasis[0], 'LG', 2, 0)
    #test_lagmatvec(lagBasis[0], lagBasis[1], 'LG', 'python', 3, 2, 0)
    #test_hmatvec(hBasis[0], hBasis[0], 'HG', 'self', 3, 1, 1)
    #test_jmatvec(jBasis[5], jBasis[5], 'JG', 'self', 1, 1)
    #test_isub(((cbases.ShenNeumann, 0), (cbases.ShenDirichlet, 1)), cmatrices.CSNSDmat, 'GC')
    #test_add(((cbases.Orthogonal, 0), (cbases.ShenDirichlet, 1)), cmatrices.CTSDmat, 'GC')
    #test_blockmatrix((D, F2, F))
    #test_sub(*mats_and_quads[15])
    #test_mul2()
    #test_div2(cBasis[1], 'GC')
    #test_helmholtz2D('chebyshev', 1)
    #test_helmholtz3D('chebyshev', 0)
    #test_biharmonic3D('chebyshev', 0)
    #test_biharmonic2D('jacobi', 0)
    #test_stencil(cbases.Phi6, cbases.Compact3)
    test_adaptive_quadrature(lbases.Phi2, lbases.ShenNeumann)
