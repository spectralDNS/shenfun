import pytest
from shenfun.chebyshev import bases as cbases
from shenfun.legendre import bases as lbases

from shenfun.la import TDMA
from shenfun import inner_product
from scipy.linalg import solve
from sympy import chebyshevt, Symbol, sin, cos, pi, lambdify
import numpy as np
import scipy.sparse.linalg as la
from itertools import product

N = 32
x = Symbol("x")

cBasis = (cbases.ChebyshevBasis,
         cbases.ShenDirichletBasis,
         cbases.ShenNeumannBasis,
         cbases.ShenBiharmonicBasis)

lBasis = (lbases.LegendreBasis, lbases.ShenDirichletBasis,
          lbases.ShenBiharmonicBasis, lbases.ShenNeumannBasis)

quads = ('GC', 'GL')
lquads = ('LG',)

all_bases_and_quads = list(product(lBasis, lquads))+list(product(cBasis, quads))

@pytest.mark.parametrize('ST,quad', product(cBasis, quads))
def test_scalarproduct(ST, quad):
    """Test fast scalar product against Vandermonde computed version"""
    ST = ST(quad=quad)
    points, weights = ST.points_and_weights(N,  ST.quad)
    f = x*x+cos(pi*x)
    fl = lambdify(x, f, 'numpy')
    fj = fl(points)
    u0 = np.zeros(N)
    u1 = np.zeros(N)
    u0 = ST.scalar_product(fj, u0, True)
    u1 = ST.scalar_product(fj, u1, False)
    assert np.allclose(u1, u0)

#test_scalarproduct(cbases.ShenDirichletBasis, 'GC')

@pytest.mark.parametrize('ST,quad', all_bases_and_quads)
def test_eval(ST, quad):
    """Test eval agains fast inverse"""
    ST = ST(quad=quad)
    points, weights = ST.points_and_weights(N,  ST.quad)
    fk = np.zeros(N)
    fj = np.random.random(N)
    fk = ST.forward(fj, fk)
    fj = ST.backward(fk, fj)
    fk = ST.forward(fj, fk)
    f = ST.eval(points, fk)
    assert np.allclose(fj, f)
    fj = ST.backward(fk, fj, False)
    fk = ST.forward(fj, fk, False)
    f = ST.eval(points, fk)
    assert np.allclose(fj, f)


#test_eval(cbases.ShenDirichletBasis, 'GC')

@pytest.mark.parametrize('test,trial', product(cBasis, cBasis))
@pytest.mark.parametrize('quad', quads)
def test_massmatrices(test, trial, quad):
    test = test(quad=quad)
    trial = trial(quad=quad)

    f_hat = np.zeros(N)
    fj = np.random.random(N)
    f_hat = trial.forward(fj, f_hat)
    fj = trial.backward(f_hat, fj)

    BBD = inner_product((test, 0), (trial, 0), N)

    f_hat = trial.forward(fj, f_hat)
    u2 = np.zeros_like(f_hat)
    u2 = BBD.matvec(f_hat, u2)
    u0 = np.zeros(N)
    u0 = test.scalar_product(fj, u0)
    s = test.slice(N)

    assert np.allclose(u0[s], u2[s])

    # Multidimensional version
    fj = fj.repeat(N*N).reshape((N, N, N)) + 1j*fj.repeat(N*N).reshape((N, N, N))
    f_hat = f_hat.repeat(N*N).reshape((N, N, N)) + 1j*f_hat.repeat(N*N).reshape((N, N, N))

    u0 = np.zeros((N, N, N), dtype=np.complex)
    u0 = test.scalar_product(fj, u0)
    u2 = np.zeros_like(f_hat)
    u2 = BBD.matvec(f_hat, u2)
    assert np.linalg.norm(u2[s]-u0[s])/(N*N*N) < 1e-12

#test_massmatrices(cBasis[2], cBasis[0], 'GC')

@pytest.mark.parametrize('ST,quad', all_bases_and_quads)
@pytest.mark.parametrize('axis', (0,1,2))
def test_transforms(ST, quad, axis):
    ST = ST(quad=quad)
    points, weights = ST.points_and_weights(N,  ST.quad)
    fj = np.random.random(N)

    # Project function to space first
    f_hat = np.zeros(N)
    f_hat = ST.forward(fj, f_hat)
    fj = ST.backward(f_hat, fj)

    # Then check if transformations work as they should
    u0 = np.zeros(N)
    u1 = np.zeros(N)
    u0 = ST.forward(fj, u0)
    u1 = ST.backward(u0, u1)

    assert np.allclose(fj, u1)

    # Multidimensional version
    bc = [np.newaxis,]*3
    bc[axis] = slice(None)
    fj = np.broadcast_to(fj[bc], (N,)*3).copy()

    u00 = np.zeros_like(fj)
    u11 = np.zeros_like(fj)
    u00 = ST.forward(fj, u00, axis=axis)
    u11 = ST.backward(u00, u11, axis=axis)
    cc = [0,]*3
    cc[axis] = slice(None)
    #from IPython import embed; embed()
    assert np.allclose(fj[cc], u11[cc])

test_transforms(lbases.ShenDirichletBasis, "LG", 2)


@pytest.mark.parametrize('ST,quad', all_bases_and_quads)
@pytest.mark.parametrize('axis', (0,1,2))
def test_axis(ST, quad, axis):
    ST = ST(quad=quad)
    points, weights = ST.points_and_weights(N,  ST.quad)
    f_hat = np.random.random(N)

    B = inner_product((ST, 0), (ST, 0), N)

    c = np.zeros_like(f_hat)
    c = B.solve(f_hat, c)

    # Multidimensional version
    bc = [np.newaxis,]*3
    bc[axis] = slice(None)
    fk = np.broadcast_to(f_hat[bc], (N,)*3).copy()
    #ck = np.zeros_like(fk)
    ck = B.solve(fk, axis=axis)
    cc = [0,]*3
    cc[axis] = slice(None)
    assert np.allclose(ck[cc], c)

#test_axis(cbases.ShenNeumannBasis, "GC", 2)


@pytest.mark.parametrize('quad', quads)
def test_CDDmat(quad):
    SD = cbases.ShenDirichletBasis(quad=quad)
    M = 128
    u = (1-x**2)*sin(np.pi*6*x)
    dudx = u.diff(x, 1)
    points, weights = SD.points_and_weights(M,  SD.quad)

    ul = lambdify(x, u, 'numpy')
    dudx_l = lambdify(x, dudx, 'numpy')
    dudx_j = dudx_l(points)
    uj = ul(points)

    dudx_j = np.zeros(M)
    u_hat = np.zeros(M)
    u_hat = SD.forward(uj, u_hat)
    uj = SD.backward(u_hat, uj)
    u_hat = SD.forward(uj, u_hat)

    uc_hat = np.zeros(M)
    uc_hat = SD.CT.forward(uj, uc_hat)
    du_hat = np.zeros(M)
    dudx_j = SD.CT.fast_derivative(uj, dudx_j)

    Cm = inner_product((SD, 0), (SD, 1), M)
    B = inner_product((SD, 0), (SD, 0), M)
    TDMASolver = TDMA(B)

    cs = np.zeros_like(u_hat)
    cs = Cm.matvec(u_hat, cs)

    # Should equal (but not exact so use extra resolution)
    cs2 = np.zeros(M)
    cs2 = SD.scalar_product(dudx_j, cs2)
    s = SD.slice(M)
    assert np.allclose(cs[s], cs2[s])

    cs = TDMASolver(cs)
    du = np.zeros(M)
    du = SD.backward(cs, du)

    assert np.linalg.norm(du[s]-dudx_j[s])/M < 1e-10

    # Multidimensional version
    u3_hat = u_hat.repeat(4*4).reshape((M, 4, 4)) + 1j*u_hat.repeat(4*4).reshape((M, 4, 4))
    cs = np.zeros_like(u3_hat)
    cs = Cm.matvec(u3_hat, cs)
    cs2 = np.zeros((M, 4, 4), dtype=np.complex)
    du3 = dudx_j.repeat(4*4).reshape((M, 4, 4)) + 1j*dudx_j.repeat(4*4).reshape((M, 4, 4))
    cs2 = SD.scalar_product(du3, cs2)

    assert np.allclose(cs[s], cs2[s], 1e-10)

    cs = TDMASolver(cs)
    d3 = np.zeros((M, 4, 4), dtype=np.complex)
    d3 = SD.backward(cs, d3)

    assert np.linalg.norm(du3[s]-d3[s])/(M*16) < 1e-10

#test_CDDmat('GL')

@pytest.mark.parametrize('test,trial', product(cBasis, cBasis))
def test_CXXmat(test, trial):
    test = test()
    trial = trial()

    CT = cBasis[0]()

    Cm = inner_product((test, 0), (trial, 1), N)
    S2 = Cm.trialfunction[0]
    S1 = Cm.testfunction[0]

    fj = np.random.randn(N)
    # project to S2
    f_hat = np.zeros(N)
    f_hat = S2.forward(fj, f_hat)
    fj = S2.backward(f_hat, fj)

    # Check S1.scalar_product(f) equals Cm*S2.forward(f)
    f_hat = S2.forward(fj, f_hat)
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)
    df = np.zeros(N)
    df = CT.fast_derivative(fj, df)
    cs2 = np.zeros(N)
    cs2 = S1.scalar_product(df, cs2)
    s = S1.slice(N)
    assert np.allclose(cs[s], cs2[s])

    # Multidimensional version
    f_hat = f_hat.repeat(4*4).reshape((N, 4, 4)) + 1j*f_hat.repeat(4*4).reshape((N, 4, 4))
    df = df.repeat(4*4).reshape((N, 4, 4)) + 1j*df.repeat(4*4).reshape((N, 4, 4))
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)
    cs2 = np.zeros((N, 4, 4), dtype=np.complex)
    cs2 = S1.scalar_product(df, cs2)

    assert np.allclose(cs[s], cs2[s])

#test_CXXmat(cBasis[2], cBasis[1])


@pytest.mark.parametrize('ST', cBasis[1:3])
@pytest.mark.parametrize('quad', quads)
def test_ADDmat(ST, quad):
    ST = ST(quad=quad)
    M = 2*N
    u = (1-x**2)*sin(np.pi*x)
    f = -u.diff(x, 2)
    ul = lambdify(x, u, 'numpy')
    fl = lambdify(x, f, 'numpy')
    points, weights = ST.points_and_weights(M,  quad)
    uj = ul(points)
    fj = fl(points)
    s = ST.slice(M)

    A = inner_product((ST, 0), (ST, 2), M)

    f_hat = np.zeros(M)
    f_hat = ST.scalar_product(fj, f_hat)

    # Test both solve interfaces
    c_hat = f_hat.copy()
    c_hat = A.solve(c_hat)

    u_hat = np.zeros_like(f_hat)
    u_hat = A.solve(f_hat, u_hat)

    assert np.allclose(c_hat[s], u_hat[s])

    u0 = np.zeros(M)
    u0 = ST.backward(u_hat, u0)
    assert np.allclose(u0, uj)

    u1 = np.zeros(M)
    u1 = ST.forward(uj, u1)
    c = np.zeros_like(u1)
    c = A.matvec(u1, c)
    s = ST.slice(M)
    assert np.allclose(c[s], f_hat[s])

#test_ADDmat(cbases.ShenNeumannBasis, "GL")

@pytest.mark.parametrize('quad', quads)
def test_SBBmat(quad):
    SB = cbases.ShenBiharmonicBasis(quad=quad)
    M = 72
    u = sin(4*pi*x)**2
    f = u.diff(x, 4)
    ul = lambdify(x, u, 'numpy')
    fl = lambdify(x, f, 'numpy')
    points, weights = SB.points_and_weights(M,  SB.quad)
    uj = ul(points)
    fj = fl(points)

    A = inner_product((SB, 0), (SB, 4), M)
    f_hat = np.zeros(M)
    f_hat = SB.scalar_product(fj, f_hat)
    u_hat = np.zeros(M)
    u_hat = A.solve(f_hat, u_hat)

    u0 = np.zeros(M)
    u0 = SB.backward(u_hat, u0)

    assert np.allclose(u0, uj)

    u1 = np.zeros(M)
    u1 = SB.forward(uj, u1)

    c = np.zeros_like(u1)
    c = A.matvec(u1, c)

    assert np.all(abs(c-f_hat)/c.max() < 1e-10)

    # Multidimensional
    c2 = (c.repeat(16).reshape((M, 4, 4))+1j*c.repeat(16).reshape((M, 4, 4)))
    u1 = (u1.repeat(16).reshape((M, 4, 4))+1j*u1.repeat(16).reshape((M, 4, 4)))

    c = np.zeros_like(u1)
    c = A.matvec(u1, c)

    assert np.allclose(c, c2)

#test_SBBmat('GC')

@pytest.mark.parametrize('quad', quads)
def test_ABBmat(quad):
    SB = cbases.ShenBiharmonicBasis(quad=quad)
    M = 4*N
    u = sin(6*pi*x)**2
    f = u.diff(x, 2)
    fl = lambdify(x, f, "numpy")
    ul = lambdify(x, u, "numpy")

    points, weights = SB.points_and_weights(M,  SB.quad)
    uj = ul(points)
    fj = fl(points)

    A = inner_product((SB, 0), (SB, 2), M)

    f_hat = np.zeros(M)
    f_hat = SB.scalar_product(fj, f_hat)
    u_hat = np.zeros(M)
    u_hat[:-4] = la.spsolve(A.diags(), f_hat[:-4])

    u0 = np.zeros(M)
    u0 = SB.backward(u_hat, u0)
    assert np.allclose(u0, uj)

    u1 = np.zeros(M)
    u1 = SB.forward(uj, u1)
    c = np.zeros_like(u1)
    c = A.matvec(u1, c)

    assert np.allclose(c, f_hat, 1e-6, 1e-6)

    # Multidimensional
    f_hat = (f_hat.repeat(16).reshape((M, 4, 4))+1j*f_hat.repeat(16).reshape((M, 4, 4)))
    u1 = (u1.repeat(16).reshape((M, 4, 4))+1j*u1.repeat(16).reshape((M, 4, 4)))

    c = np.zeros_like(u1)
    c = A.matvec(u1, c)

    assert np.allclose(c, f_hat, 1e-6, 1e-6)

    B = inner_product((SB, 0), (SB, 0), M)

    u0 = np.random.randn(M)
    u0_hat = np.zeros(M)
    u0_hat = SB.forward(u0, u0_hat)
    u0 = SB.backward(u0_hat, u0)
    b = np.zeros(M)
    k = 2.
    c0 = np.zeros_like(u0_hat)
    c1 = np.zeros_like(u0_hat)
    b = A.matvec(u0_hat, c0) - k**2*B.matvec(u0_hat, c1)
    AA = A.diags().toarray() - k**2*B.diags().toarray()
    z0_hat = np.zeros(M)
    z0_hat[:-4] = solve(AA, b[:-4])
    z0 = np.zeros(M)
    z0 = SB.backward(z0_hat, z0)
    assert np.allclose(z0, u0)

#test_ABBmat('GC')
