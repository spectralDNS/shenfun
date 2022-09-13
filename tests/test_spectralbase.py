from itertools import product
import pytest
from shenfun import FunctionSpace, TrialFunction, inner, Array, \
    Function, TestFunction as _TestFunction
import numpy as np
import sympy as sp

x = sp.symbols('x', real=True, positive=True)
rv = (sp.cos(x), sp.sin(x))

quads = {
    'chebyshev': ('GC', 'GL'),
    'legendre': ('LG', 'GL'),
    'laguerre': ('LG',),
    'hermite': ('HG',),
    'jacobi': ('JG',),
    'ultraspherical': ('QG',),
    'chebyshevu': ('GU', 'GC')
}

def test_mesh():
    N = 4
    F = FunctionSpace(N, 'F', dtype='d', coordinates=((x,), rv))
    xj = F.mesh()
    xx = F.cartesian_mesh()
    assert np.sum(abs(xx[0] - np.cos(xj))*abs(xx[1] - np.sin(xj))) < 1e-12

def test_bcs():
    F = FunctionSpace(8, 'C', bc='u(-1)=1&&u(1)=1')
    assert str(F.bc.bc) == "{'left': {'D': 1.0}, 'right': {'D': 1.0}}"
    F = FunctionSpace(8, 'C', bc="u'(-1)=1&&u'(1)=1")
    assert str(F.bc.bc) == "{'left': {'N': 1.0}, 'right': {'N': 1.0}}"
    F = FunctionSpace(8, 'C', bc="u''(-1)=1&&u'''(1)=1")
    assert str(F.bc.bc) == "{'left': {'N2': 1.0}, 'right': {'N3': 1.0}}"
    F = FunctionSpace(8, 'C', bc="u''''(-1)=1")
    assert str(F.bc.bc) == "{'left': {'N4': 1.0}, 'right': {}}"

@pytest.mark.parametrize('family', 'CLUQ')
def test_inner_kind(family):
    N = 8
    f = sp.cos(8*sp.pi*x)
    B = FunctionSpace(8, family)
    v = _TestFunction(B)
    ff = inner(v, f)
    fv = inner(v, f, kind='vandermonde')
    fr = inner(v, f, kind='recursive')
    assert np.linalg.norm(ff-fv) < 1e-12
    assert np.linalg.norm(ff-fr) < 1e-12

    fj = Array(B, buffer=f)
    ff = fj.forward()
    fv = fj.forward(kind='vandermonde')
    fr = fj.forward(kind='recursive')
    assert np.linalg.norm(ff-fv) < 1e-12
    assert np.linalg.norm(ff-fr) < 1e-12

    fh = Function(B)
    fh[1] = 1
    assert np.linalg.norm(fh.backward(mesh='quadrature')-B.evaluate_basis(B.mesh(kind='quadrature'), 1)) < 1e-12
    assert np.linalg.norm(fh.backward(mesh='uniform')-B.evaluate_basis(B.mesh(kind='uniform'), 1)) < 1e-12
    ff = fh.backward()
    fv = fh.backward(kind='vandermonde')
    fr = fh.backward(kind='recursive')
    assert np.linalg.norm(ff-fv) < 1e-12
    assert np.linalg.norm(ff-fr) < 1e-12

@pytest.mark.parametrize('family', 'CLUQ')
def test_new(family):
    N = 8
    D = FunctionSpace(N, family, bc=(1, 1))
    D0 = D.get_homogeneous()
    assert str(D0.bc.bc) == "{'left': {'D': 0}, 'right': {'D': 0}}"
    assert D0.bc.bcs == [0, 0]
    assert D.family() == {'C': 'chebyshev', 'L': 'legendre', 'U': 'chebyshevu', 'Q': 'ultraspherical'}[family]

    D = FunctionSpace(0, family)
    A = D.get_adaptive(x**4)
    assert A.N == 5
    assert A.is_orthogonal == True
    T = Function(D, buffer=x**4)
    assert len(T) == 5

    R = D.get_refined(10)
    assert R.N == 10
    assert R.is_orthogonal == True

    C = FunctionSpace(8, family, domain=(0, 2), coordinates=((x,), (x, x**2)))
    measure = C.coors.get_sqrt_det_g()
    y = measure.free_symbols.pop()
    xj, wj = C.points_and_weights()
    assert np.allclose(C.get_measured_array(xj.copy()), xj*sp.lambdify(y, measure)(C.mesh()))

def test_sympy_stencil():
    D = FunctionSpace(8, 'L', bc='u(-1)=0&&u(1)=0', scaled=True)
    i, j = sp.symbols('i,j', integer=True)
    assert str(D.sympy_stencil()) == 'KroneckerDelta(i, j)/sqrt(4*i + 6) - KroneckerDelta(j, i + 2)/sqrt(4*i + 6)'
    assert str(D.sympy_stencil(implicit='a')) == 'KroneckerDelta(i, j)*a0(i) + KroneckerDelta(j, i + 2)*a2(i)'

def test_map_domain():
    B = FunctionSpace(8, 'C', domain=(0, 2))
    assert B.map_reference_domain(1) == 0
    assert B.map_reference_domain(0) == -1
    assert np.alltrue(B.map_reference_domain(np.array([0.5, 1.5])) == np.array([-0.5, 0.5]))
    assert B.map_true_domain(0) == 1.0
    assert np.alltrue(B.map_true_domain(np.array([-1, 1])) == np.array([0, 2]))
    assert B.map_expression_true_domain(x) == x+1
    assert B.domain_length() == 2
    assert B.dims() == (8,)
    assert B.dtype == np.dtype('d')
    assert B.is_padded == False
    assert B.ndim == 1
    xj, wj = B.points_and_weights()
    assert np.alltrue(B.get_measured_weights(measure=x) == xj*wj)
    assert np.alltrue(B.get_measured_weights(measure=2) == 2*wj)

@pytest.mark.parametrize('family', 'CLUQJ')
def test_constant_inner(family):
    D = FunctionSpace(6, family, alpha=1, beta=2)
    for quad in quads[D.family()]:
        q = inner(1, Array(D, buffer=x**2))
        assert abs(q-2/3) < 1e-8
