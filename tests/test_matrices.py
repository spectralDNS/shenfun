import pytest
import numpy as np
from shenfun.chebyshev import matrices as cmatrices
from shenfun.legendre import matrices as lmatrices
from copy import deepcopy
import six
from itertools import product

cmats = list(filter(lambda f: f.endswith('mat'), vars(cmatrices).keys()))
cmats = ['.'.join(('cmatrices', m)) for m in cmats]
cmats.remove('cmatrices.mat')
lmats = list(filter(lambda f: f.endswith('mat'), vars(lmatrices).keys()))
lmats = ['.'.join(('lmatrices', m)) for m in lmats]
lmats.remove('lmatrices.mat')

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

all_mats_and_quads = list(product(lmats, lquads))+list(product(cmats, cquads))

@pytest.mark.parametrize('mat,quad', all_mats_and_quads)
def test_mat(mat, quad):
    """Test that matrix equals one that is automatically created"""
    mat = eval(mat)
    mat.testfunction[0].quad = quad
    m = mat(k)
    m.test_sanity()

@pytest.mark.parametrize('mat,quad', all_mats_and_quads)
@pytest.mark.parametrize('format', formats)
def test_matvec(mat, quad, format):
    """Test matrix-vector product"""
    global c, c1, d, d1, k
    mat = eval(mat)
    mat.testfunction[0].quad = quad
    m = mat(k)
    c = m.matvec(a, c, format='csr')
    c1 = m.matvec(a, c1, format=format)
    assert np.allclose(c, c1)

    d.fill(0)
    d1.fill(0)
    d = m.matvec(b, d, format='csr')
    d1 = m.matvec(b, d1, format=format)
    assert np.allclose(d, d1)

##test_matvec('cmatrices.CDNmat', 'cython', 'GL')

@pytest.mark.parametrize('mat,quad', all_mats_and_quads)
def test_imul(mat, quad):
    mat = eval(mat)
    mat.testfunction[0].quad = quad
    m = mat(k)
    mc = deepcopy(dict(m))
    m *= 2
    for key, val in six.iteritems(m):
        assert np.allclose(val, mc[key]*2)

#test_imul('BDNmat', 'GL')

@pytest.mark.parametrize('mat,quad', all_mats_and_quads)
def test_mul(mat, quad):
    mat = eval(mat)
    mat.testfunction[0].quad = quad
    m = mat(k)
    mc = 2.*m
    for key, val in six.iteritems(mc):
        assert np.allclose(val, m[key]*2.)

@pytest.mark.parametrize('mat,quad', all_mats_and_quads)
def test_rmul(mat, quad):
    mat = eval(mat)
    mat.testfunction[0].quad = quad
    m = mat(k)
    mc = m*2.
    for key, val in six.iteritems(mc):
        assert np.allclose(val, m[key]*2.)

@pytest.mark.parametrize('mat,quad', all_mats_and_quads)
def test_div(mat, quad):
    mat = eval(mat)
    mat.testfunction[0].quad = quad
    m = mat(k)
    mc = m/2.
    for key, val in six.iteritems(mc):
        assert np.allclose(val, m[key]/2.)

@pytest.mark.parametrize('mat,quad', all_mats_and_quads)
def test_add(mat, quad):
    mat = eval(mat)
    mat.testfunction[0].quad = quad
    m = mat(k)
    mc = m + m
    for key, val in six.iteritems(mc):
        assert np.allclose(val, m[key]*2)

@pytest.mark.parametrize('mat,quad', all_mats_and_quads)
def test_iadd(mat, quad):
    mat = eval(mat)
    mat.testfunction[0].quad = quad
    m = mat(k)
    mc = deepcopy(m)
    m += mc
    for key, val in six.iteritems(m):
        assert np.allclose(val, mc[key]*2)

@pytest.mark.parametrize('mat,quad', all_mats_and_quads)
def test_isub(mat, quad):
    mat = eval(mat)
    mat.testfunction[0].quad = quad
    m = mat(k)
    mc = deepcopy(m)
    m -= mc
    for key, val in six.iteritems(m):
        assert np.allclose(val, 0)

@pytest.mark.parametrize('mat,quad', all_mats_and_quads)
def test_sub(mat, quad):
    mat = eval(mat)
    mat.testfunction[0].quad = quad
    m = mat(k)
    mc = m - m
    for key, val in six.iteritems(mc):
        assert np.allclose(val, 0)
