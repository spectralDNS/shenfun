import pytest
import numpy as np
from scipy.sparse.linalg import spsolve
import shenfun
from shenfun.chebyshev import matrices as cmatrices
from shenfun.chebyshev import bases as cbases
from shenfun.legendre import matrices as lmatrices
from shenfun.legendre import bases as lbases
from shenfun.chebyshev import la as cla
from shenfun.legendre import la as lla
from copy import copy
import mpi4py_fft
from mpi4py import MPI
import six
from itertools import product

cBasis = (cbases.Basis,
          cbases.ShenDirichletBasis,
          cbases.ShenNeumannBasis,
          cbases.ShenBiharmonicBasis)

lBasis = (lbases.Basis,
          lbases.ShenDirichletBasis,
          lbases.ShenBiharmonicBasis,
          lbases.ShenNeumannBasis)

cquads = ('GC', 'GL')
lquads = ('LG',)
formats = ('dia', 'cython', 'python', 'self')

N = 16
k = np.arange(N).astype(float)
a = np.random.random(N)
b = np.random.random((N, N, N))
c = np.zeros(N)
c1= np.zeros(N)
d = np.zeros((N, N, N))
d1 = np.zeros((N, N, N))

cbases2 = list(product(cBasis, cBasis))
lbases2 = list(product(lBasis, lBasis))
bases2 = cbases2+lbases2

cmats_and_quads = [list(k[0])+[k[1]] for k in product([(k, v) for k, v in six.iteritems(cmatrices.mat)], cquads)]
lmats_and_quads = [list(k[0])+[k[1]] for k in product([(k, v) for k, v in six.iteritems(lmatrices.mat)], lquads)]
mats_and_quads = cmats_and_quads+lmats_and_quads

#cmats_and_quads_ids = ['-'.join(i) for i in product([v.__name__ for v in cmatrices.mat.values()], cquads)]
#lmats_and_quads_ids = ['-'.join(i) for i in product([v.__name__ for v in lmatrices.mat.values()], lquads)]

@pytest.mark.parametrize('key, mat, quad', mats_and_quads)
def test_mat(key, mat, quad):
    """Test that matrices built by hand equals those automatically generated"""
    test = key[0]
    trial = key[1]
    testfunction = (test[0](N, quad=quad), test[1])
    trialfunction = (trial[0](N, quad=quad), trial[1])
    mat = mat(testfunction, trialfunction)
    shenfun.check_sanity(mat, testfunction, trialfunction)

@pytest.mark.parametrize('b0,b1', cbases2)
@pytest.mark.parametrize('quad', cquads)
@pytest.mark.parametrize('format', formats)
@pytest.mark.parametrize('axis', (0,1,2))
@pytest.mark.parametrize('k', range(5))
def test_cmatvec(b0, b1, quad, format, axis, k):
    """Test matrix-vector product"""
    global c, c1, d, d1
    b0 = b0(N, quad=quad)
    b1 = b1(N, quad=quad)
    mat = shenfun.spectralbase.inner_product((b0, 0), (b1, k))
    c = mat.matvec(a, c, format='csr')
    c1 = mat.matvec(a, c1, format=format)
    assert np.allclose(c, c1)

    d.fill(0)
    d1.fill(0)
    d = mat.matvec(b, d, format='csr', axis=axis)
    d1 = mat.matvec(b, d1, format=format, axis=axis)
    assert np.allclose(d, d1)

    # Test multidimensional with axis equals 1D case
    d1.fill(0)
    bc = [np.newaxis,]*3
    bc[axis] = slice(None)
    fj = np.broadcast_to(a[tuple(bc)], (N,)*3).copy()
    d1 = mat.matvec(fj, d1, format=format, axis=axis)
    cc = [0,]*3
    cc[axis] = slice(None)
    assert np.allclose(c, d1[tuple(cc)])

@pytest.mark.parametrize('b0,b1', lbases2)
@pytest.mark.parametrize('quad', lquads)
@pytest.mark.parametrize('format', formats)
@pytest.mark.parametrize('axis', (0,1,2))
@pytest.mark.parametrize('k0,k1', product((0,1,2), (0,1,2)))
def test_lmatvec(b0, b1, quad, format, axis, k0, k1):
    """Test matrix-vector product"""
    global c, c1, d, d1
    b0 = b0(N, quad=quad)
    b1 = b1(N, quad=quad)
    mat = shenfun.spectralbase.inner_product((b0, k0), (b1, k1))
    c = mat.matvec(a, c, format='csr')
    c1 = mat.matvec(a, c1, format=format)
    assert np.allclose(c, c1)

    d.fill(0)
    d1.fill(0)
    d = mat.matvec(b, d, format='csr', axis=axis)
    d1 = mat.matvec(b, d1, format=format, axis=axis)
    assert np.allclose(d, d1)

    # Test multidimensional with axis equals 1D case
    d1.fill(0)
    bc = [np.newaxis,]*3
    bc[axis] = slice(None)
    fj = np.broadcast_to(a[tuple(bc)], (N,)*3).copy()
    d1 = mat.matvec(fj, d1, format=format, axis=axis)
    cc = [0,]*3
    cc[axis] = slice(None)
    assert np.allclose(c, d1[tuple(cc)])

@pytest.mark.parametrize('key, mat, quad', mats_and_quads)
def test_imul(key, mat, quad):
    test = key[0]
    trial = key[1]
    mat = mat((test[0](N, quad=quad), test[1]),
              (trial[0](N, quad=quad), trial[1]))
    mc = copy(dict(mat))
    mat *= 2
    assert mat.scale == 2.0

    mat = shenfun.SparseMatrix(copy(dict(mc)), mat.shape)
    mat *= 2
    assert mat.scale == 2.0

@pytest.mark.parametrize('key, mat, quad', mats_and_quads)
def test_mul(key, mat, quad):
    test = key[0]
    trial = key[1]
    m = mat((test[0](N, quad=quad), test[1]),
            (trial[0](N, quad=quad), trial[1]))
    mc = 2.*m
    assert mc.scale == 2.0

    mat = shenfun.SparseMatrix(copy(dict(m)), m.shape)
    mc = 2.*mat
    assert mc.scale == 2.0

def test_mul2():
    mat = shenfun.SparseMatrix({0: 1}, (3, 3))
    v = np.ones(3)
    c = mat * v
    assert np.allclose(c, 1)
    mat = shenfun.SparseMatrix({-2:1, -1:1, 0: 1, 1:1, 2:1}, (3, 3))
    c = mat * v
    assert np.allclose(c, 3)
    SD = shenfun.Basis(8, "L", bc=(0, 0), scaled=True)
    u = shenfun.TrialFunction(SD)
    v = shenfun.TestFunction(SD)
    mat = shenfun.inner(shenfun.grad(u), shenfun.grad(v))
    z = shenfun.Function(SD, val=1)
    c = mat * z
    assert np.allclose(c[:6], 1)

@pytest.mark.parametrize('key, mat, quad', mats_and_quads)
def test_rmul(key, mat, quad):
    test = key[0]
    trial = key[1]
    m = mat((test[0](N, quad=quad), test[1]),
            (trial[0](N, quad=quad), trial[1]))
    mc = m*2.
    assert mc.scale == 2.0

    mat = shenfun.SparseMatrix(copy(dict(m)), m.shape)
    mc = mat*2.
    assert mc.scale == 2.0

@pytest.mark.parametrize('key, mat, quad', mats_and_quads)
def test_div(key, mat, quad):
    test = key[0]
    trial = key[1]
    m = mat((test[0](N, quad=quad), test[1]),
            (trial[0](N, quad=quad), trial[1]))

    mc = m/2.
    assert mc.scale == 0.5

    mat = shenfun.SparseMatrix(copy(dict(m)), m.shape)
    mc = mat/2.
    assert mc.scale == 0.5

@pytest.mark.parametrize('basis, quad', list(product(cBasis, cquads))+
                         list(product(lBasis, lquads)))
def test_div2(basis, quad):
    B = basis(8, quad=quad)
    u = shenfun.TrialFunction(B)
    v = shenfun.TestFunction(B)
    m = shenfun.inner(u, v)
    z = shenfun.Function(B, val=1)
    c = m / z
    m2 = m.diags('csr')
    c2 = spsolve(m2, z[B.slice()])
    assert np.allclose(c2, c[B.slice()])

@pytest.mark.parametrize('key, mat, quad', mats_and_quads)
def test_add(key, mat, quad):
    test = key[0]
    trial = key[1]
    m = mat((test[0](N, quad=quad), test[1]),
            (trial[0](N, quad=quad), trial[1]))
    mc = m + m
    assert mc.scale == 2.0

    mat = shenfun.SparseMatrix(copy(dict(m)), m.shape)
    mc = m + mat
    for key, val in six.iteritems(mc):
        assert np.allclose(val, m[key]*2)

@pytest.mark.parametrize('key, mat, quad', mats_and_quads)
def test_iadd(key, mat, quad):
    test = key[0]
    trial = key[1]
    m = mat((test[0](N, quad=quad), test[1]),
            (trial[0](N, quad=quad), trial[1]))
    mc = copy(m)
    m += mc
    assert m.scale == 2.0

    m1 = shenfun.SparseMatrix(copy(dict(m)), m.shape)
    m2 = shenfun.SparseMatrix(copy(dict(m)), m.shape)
    m1 += m2
    assert m1.scale == 2.0

@pytest.mark.parametrize('key, mat, quad', mats_and_quads)
def test_isub(key, mat, quad):
    test = key[0]
    trial = key[1]
    m = mat((test[0](N, quad=quad), test[1]),
            (trial[0](N, quad=quad), trial[1]))
    mc = copy(m)
    m -= mc
    assert m.scale == 0.0

    m1 = shenfun.SparseMatrix(copy(dict(m)), m.shape)
    m2 = shenfun.SparseMatrix(copy(dict(m)), m.shape)
    m1 -= m2
    assert m1.scale == 0.0

@pytest.mark.parametrize('key, mat, quad', mats_and_quads)
def test_sub(key, mat, quad):
    test = key[0]
    trial = key[1]
    m = mat((test[0](N, quad=quad), test[1]),
            (trial[0](N, quad=quad), trial[1]))
    mc = m - m
    assert mc.scale == 0.0

    m1 = shenfun.SparseMatrix(copy(dict(m)), m.shape)
    m2 = shenfun.SparseMatrix(copy(dict(m)), m.shape)

    mc = m1 - m2
    assert mc.scale == 0.0

allaxes2D = {0: (0, 1), 1: (1, 0)}
allaxes3D = {0: (0, 1, 2), 1: (1, 0, 2), 2: (2, 0, 1)}

@pytest.mark.parametrize('axis', (0, 1, 2))
@pytest.mark.parametrize('family', ('chebyshev','legendre'))
def test_helmholtz3D(family, axis):
    la = lla
    if family == 'chebyshev':
        la = cla
    N = (8, 9, 10)
    SD = shenfun.Basis(N[allaxes3D[axis][0]], family=family, bc=(0, 0))
    K1 = shenfun.Basis(N[allaxes3D[axis][1]], family='F', dtype='D')
    K2 = shenfun.Basis(N[allaxes3D[axis][2]], family='F', dtype='d')
    subcomms = mpi4py_fft.pencil.Subcomm(MPI.COMM_WORLD, [0, 1, 1])
    bases = [0]*3
    bases[allaxes3D[axis][0]] = SD
    bases[allaxes3D[axis][1]] = K1
    bases[allaxes3D[axis][2]] = K2
    T = shenfun.TensorProductSpace(subcomms, bases, axes=allaxes3D[axis])
    u = shenfun.TrialFunction(T)
    v = shenfun.TestFunction(T)
    if family == 'chebyshev':
        mat = shenfun.inner(v, shenfun.div(shenfun.grad(u)))
    else:
        mat = shenfun.inner(shenfun.grad(v), shenfun.grad(u))

    H = la.Helmholtz(**mat)
    u = shenfun.Function(T)
    u[:] = np.random.random(u.shape) + 1j*np.random.random(u.shape)
    f = shenfun.Function(T)
    f = H.matvec(u, f, axis=axis)

    g0 = shenfun.Function(T)
    g1 = shenfun.Function(T)
    g0 = mat['ADDmat'].matvec(u, g0, axis=axis)
    g1 = mat['BDDmat'].matvec(u, g1, axis=axis)

    assert np.linalg.norm(f-(g0+g1)) < 1e-12


@pytest.mark.parametrize('axis', (0, 1))
@pytest.mark.parametrize('family', ('chebyshev','legendre'))
def test_helmholtz2D(family, axis):
    la = lla
    if family == 'chebyshev':
        la = cla
    N = (8, 9)
    SD = shenfun.Basis(N[axis], family=family, bc=(0, 0))
    K1 = shenfun.Basis(N[(axis+1)%2], family='F', dtype='d')
    subcomms = mpi4py_fft.pencil.Subcomm(MPI.COMM_WORLD, [0, 1])
    bases = [K1]
    bases.insert(axis, SD)
    T = shenfun.TensorProductSpace(subcomms, bases, axes=allaxes2D[axis])
    u = shenfun.TrialFunction(T)
    v = shenfun.TestFunction(T)
    if family == 'chebyshev':
        mat = shenfun.inner(v, shenfun.div(shenfun.grad(u)))
    else:
        mat = shenfun.inner(shenfun.grad(v), shenfun.grad(u))

    H = la.Helmholtz(**mat)
    u = shenfun.Function(T)
    u[:] = np.random.random(u.shape) + 1j*np.random.random(u.shape)
    f = shenfun.Function(T)
    f = H.matvec(u, f, axis=axis)

    g0 = shenfun.Function(T)
    g1 = shenfun.Function(T)
    g0 = mat['ADDmat'].matvec(u, g0, axis=axis)
    g1 = mat['BDDmat'].matvec(u, g1, axis=axis)

    assert np.linalg.norm(f-(g0+g1)) < 1e-12

@pytest.mark.parametrize('axis', (0, 1, 2))
@pytest.mark.parametrize('family', ('chebyshev','legendre'))
def test_biharmonic3D(family, axis):
    la = lla
    if family == 'chebyshev':
        la = cla
    N = (16, 16, 16)
    SD = shenfun.Basis(N[allaxes3D[axis][0]], family=family, bc='Biharmonic')
    K1 = shenfun.Basis(N[allaxes3D[axis][1]], family='F', dtype='D')
    K2 = shenfun.Basis(N[allaxes3D[axis][2]], family='F', dtype='d')
    subcomms = mpi4py_fft.pencil.Subcomm(MPI.COMM_WORLD, [0, 1, 1])
    bases = [0]*3
    bases[allaxes3D[axis][0]] = SD
    bases[allaxes3D[axis][1]] = K1
    bases[allaxes3D[axis][2]] = K2
    T = shenfun.TensorProductSpace(subcomms, bases, axes=allaxes3D[axis])
    u = shenfun.TrialFunction(T)
    v = shenfun.TestFunction(T)
    if family == 'chebyshev':
        mat = shenfun.inner(v, shenfun.div(shenfun.grad(shenfun.div(shenfun.grad(u)))))
    else:
        mat = shenfun.inner(shenfun.div(shenfun.grad(v)), shenfun.div(shenfun.grad(u)))

    H = la.Biharmonic(**mat)
    u = shenfun.Function(T)
    u[:] = np.random.random(u.shape) + 1j*np.random.random(u.shape)
    f = shenfun.Function(T)
    f = H.matvec(u, f, axis=axis)

    g0 = shenfun.Function(T)
    g1 = shenfun.Function(T)
    g2 = shenfun.Function(T)
    amat = 'ABBmat' if family == 'chebyshev' else 'PBBmat'
    g0 = mat['SBBmat'].matvec(u, g0, axis=axis)
    g1 = mat[amat].matvec(u, g1, axis=axis)
    g2 = mat['BBBmat'].matvec(u, g2, axis=axis)

    assert np.linalg.norm(f-(g0+g1+g2)) < 1e-8

@pytest.mark.parametrize('axis', (0, 1))
@pytest.mark.parametrize('family', ('chebyshev','legendre'))
def test_biharmonic2D(family, axis):
    la = lla
    if family == 'chebyshev':
        la = cla
    N = (16, 16)
    SD = shenfun.Basis(N[axis], family=family, bc='Biharmonic')
    K1 = shenfun.Basis(N[(axis+1)%2], family='F', dtype='d')
    subcomms = mpi4py_fft.pencil.Subcomm(MPI.COMM_WORLD, (0, 1))
    bases = [K1]
    bases.insert(axis, SD)
    T = shenfun.TensorProductSpace(subcomms, bases, axes=allaxes2D[axis])
    u = shenfun.TrialFunction(T)
    v = shenfun.TestFunction(T)
    if family == 'chebyshev':
        mat = shenfun.inner(v, shenfun.div(shenfun.grad(shenfun.div(shenfun.grad(u)))))
    else:
        mat = shenfun.inner(shenfun.div(shenfun.grad(v)), shenfun.div(shenfun.grad(u)))

    H = la.Biharmonic(**mat)
    u = shenfun.Function(T)
    u[:] = np.random.random(u.shape) + 1j*np.random.random(u.shape)
    f = shenfun.Function(T)
    f = H.matvec(u, f, axis=axis)

    g0 = shenfun.Function(T)
    g1 = shenfun.Function(T)
    g2 = shenfun.Function(T)
    amat = 'ABBmat' if family == 'chebyshev' else 'PBBmat'
    g0 = mat['SBBmat'].matvec(u, g0, axis=axis)
    g1 = mat[amat].matvec(u, g1, axis=axis)
    g2 = mat['BBBmat'].matvec(u, g2, axis=axis)

    assert np.linalg.norm(f-(g0+g1+g2)) < 1e-8


if __name__ == '__main__':
    #test_add(*mats_and_quads[0])
    #test_mul2()
    #test_div2(cBasis[0], 'GC')
    test_helmholtz3D('legendre', 2)
    #test_helmholtz2D('legendre', 0)
    #test_biharmonic3D('chebyshev', 2)
    #test_biharmonic2D('chebyshev', 0)
