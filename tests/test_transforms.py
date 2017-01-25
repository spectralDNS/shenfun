import pytest
from shenfun.chebyshev import bases as cbases

from shenfun.chebyshev.matrices import BNNmat, BTTmat, BDDmat, CDDmat, CDNmat, \
    BNDmat, CNDmat, BDNmat, ADDmat, ANNmat, CTDmat, BDTmat, CDTmat, BTDmat, \
    BTNmat, BBBmat, ABBmat, SBBmat, CDBmat, CBDmat, ATTmat, BBDmat
from shenfun.la import TDMA
from scipy.linalg import solve
from sympy import chebyshevt, Symbol, sin, cos, pi
import numpy as np
import scipy.sparse.linalg as la

N = 32
x = Symbol("x")

Basis = (cbases.ChebyshevBasis,
         cbases.ShenDirichletBasis,
         cbases.ShenNeumannBasis,
         cbases.ShenBiharmonicBasis)

quads = ('GC', 'GL')

@pytest.mark.parametrize('ST', Basis)
@pytest.mark.parametrize('quad', quads)
def test_scalarproduct(ST, quad):
    """Test fast scalar product against Vandermonde computed version"""
    ST = ST(quad=quad)
    points, weights = ST.points_and_weights(N,  ST.quad)
    f = x*x+cos(pi*x)
    fj = np.array([f.subs(x, j) for j in points], dtype=float)
    u0 = np.zeros(N)
    u1 = np.zeros(N)
    ST.fast_transform = True
    u0 = ST.scalar_product(fj, u0)
    ST.fast_transform = False
    u1 = ST.scalar_product(fj, u1)
    assert np.allclose(u1, u0)


@pytest.mark.parametrize('ST', Basis)
@pytest.mark.parametrize('quad', quads)
def test_BNNmat(ST, quad):
    ST = ST(quad=quad)
    points, weights = ST.points_and_weights(N,  ST.quad)
    f_hat = np.zeros(N)
    fj = np.random.random(N)
    u0 = np.zeros(N)
    B = ST.get_mass_matrix()(np.arange(N).astype(np.float), ST.quad)
    f_hat = ST.forward(fj, f_hat)
    fj = ST.backward(f_hat, fj)
    u0 = ST.scalar_product(fj, u0)
    f_hat = ST.forward(fj, f_hat)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)

    assert np.allclose(u2[:-2], u0[:-2])

    # Multidimensional version
    fj = fj.repeat(16).reshape((N, 4, 4)) + 1j*fj.repeat(16).reshape((N, 4, 4))
    u0 = np.zeros((N, 4, 4), dtype=np.complex)
    f_hat = np.zeros((N, 4, 4), dtype=np.complex)
    u0 = ST.scalar_product(fj, u0)
    f_hat = ST.forward(fj, f_hat)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)
    assert np.allclose(u2[:-2], u0[:-2])

#test_BNNmat(ShenNeumannBasis, 'GC')

@pytest.mark.parametrize('quad', quads)
@pytest.mark.parametrize('mat', (BNDmat, BDNmat))
def test_BDNmat(mat, quad):
    B = mat(np.arange(N).astype(np.float), quad)
    S2 = B.trialfunction
    S1 = B.testfunction

    f_hat = np.zeros(N)
    fj = np.random.random(N)
    f_hat = S2.forward(fj, f_hat)
    fj = S2.backward(f_hat, fj)

    f_hat = S2.forward(fj, f_hat)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)
    u0 = np.zeros(N)
    u0 = S1.scalar_product(fj, u0)
    #from IPython import embed; embed()
    assert np.allclose(u0, u2)

    # Multidimensional version
    fj = fj.repeat(16).reshape((N, 4, 4)) + 1j*fj.repeat(16).reshape((N, 4, 4))
    f_hat = f_hat.repeat(16).reshape((N, 4, 4)) + 1j*f_hat.repeat(16).reshape((N, 4, 4))

    u0 = np.zeros((N, 4, 4), dtype=np.complex)
    u0 = S1.scalar_product(fj, u0)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)
    assert np.linalg.norm(u2-u0)/(N*16) < 1e-12

#test_BDNmat(BNDmat, "GL")

@pytest.mark.parametrize('quad', quads)
def test_BDTmat(quad):
    SD = cbases.ShenDirichletBasis(quad=quad)
    ST = cbases.ChebyshevBasis(quad=quad)

    f_hat = np.zeros(N)
    fj = np.random.random(N)
    f_hat = ST.forward(fj, f_hat)
    fj = ST.backward(f_hat, fj)

    B = BDTmat(np.arange(N).astype(np.float), SD.quad)

    f_hat = ST.forward(fj, f_hat)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)
    u2 = B.matvec(f_hat, u2, 'csr')
    u2 = B.matvec(f_hat, u2, 'csc')
    u2 = B.matvec(f_hat, u2, 'dia')
    u0 = np.zeros(N)
    u0 = SD.scalar_product(fj, u0)

    #from IPython import embed; embed()
    assert np.allclose(u0, u2)

    # Multidimensional version
    fj = fj.repeat(16).reshape((N, 4, 4)) + 1j*fj.repeat(16).reshape((N, 4, 4))
    f_hat = f_hat.repeat(16).reshape((N, 4, 4)) + 1j*f_hat.repeat(16).reshape((N, 4, 4))

    u0 = np.zeros((N, 4, 4), dtype=np.complex)
    u0 = SD.scalar_product(fj, u0)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)
    assert np.linalg.norm(u2-u0)/(N*16) < 1e-12

#test_BDTmat((ShenDirichletBasis("GL"), ShenNeumannBasis("GL")))

@pytest.mark.parametrize('quad', quads)
def test_BBDmat(quad):
    SB = cbases.ShenBiharmonicBasis(quad=quad)
    SD = cbases.ShenDirichletBasis(quad=quad)

    f_hat = np.zeros(N)
    fj = np.random.random(N)
    f_hat = SD.forward(fj, f_hat)
    fj = SD.backward(f_hat, fj)

    B = BBDmat(np.arange(N).astype(np.float), SB.quad)

    f_hat = SD.forward(fj, f_hat)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)
    u0 = np.zeros(N)
    u0 = SB.scalar_product(fj, u0)

    assert np.allclose(u0, u2)

    # Multidimensional version
    fj = fj.repeat(N*N).reshape((N, N, N)) + 1j*fj.repeat(N*N).reshape((N, N, N))
    f_hat = f_hat.repeat(N*N).reshape((N, N, N)) + 1j*f_hat.repeat(N*N).reshape((N, N, N))

    u0 = np.zeros((N, N, N), dtype=np.complex)
    u0 = SB.scalar_product(fj, u0)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)
    assert np.linalg.norm(u2-u0)/(N*N*N) < 1e-12

#test_BBDmat((ShenBiharmonicBasis("GL"), ShenDirichletBasis("GL")))

@pytest.mark.parametrize('mat', (BTDmat, BTNmat))
@pytest.mark.parametrize('quad', quads)
def test_BTXmat(mat, quad):
    B = mat(np.arange(N).astype(np.float), quad)
    SX = B.trialfunction
    ST = B.testfunction

    f_hat = np.zeros(N)
    fj = np.random.random(N)
    f_hat = SX.forward(fj, f_hat)
    fj = SX.backward(f_hat, fj)

    f_hat = SX.forward(fj, f_hat)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)
    u0 = np.zeros(N)
    u0 = ST.scalar_product(fj, u0)

    #from IPython import embed; embed()
    assert np.allclose(u0, u2)

    # Multidimensional version
    fj = fj.repeat(16).reshape((N, 4, 4)) + 1j*fj.repeat(16).reshape((N, 4, 4))
    f_hat = f_hat.repeat(16).reshape((N, 4, 4)) + 1j*f_hat.repeat(16).reshape((N, 4, 4))

    u0 = np.zeros((N, 4, 4), dtype=np.complex)
    u0 = ST.scalar_product(fj, u0)
    u2 = np.zeros_like(f_hat)
    u2 = B.matvec(f_hat, u2)
    assert np.linalg.norm(u2-u0)/(N*16) < 1e-12

#test_BTXmat(BTDmat, 'GC')

@pytest.mark.parametrize('ST', Basis)
@pytest.mark.parametrize('quad', quads)
def test_transforms(ST, quad):
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
    fj = fj.repeat(16).reshape((N, 4, 4)) + 1j*fj.repeat(16).reshape((N, 4, 4))
    u0 = np.zeros((N, 4, 4), dtype=np.complex)
    u1 = np.zeros((N, 4, 4), dtype=np.complex)
    u0 = ST.forward(fj, u0)
    u1 = ST.backward(u0, u1)

    assert np.allclose(fj, u1)

#test_transforms(ShenBiharmonicBasis("GC"))

@pytest.mark.parametrize('quad', quads)
def test_CDDmat(quad):
    SD = cbases.ShenDirichletBasis(quad=quad)
    M = 256
    u = (1-x**2)*sin(np.pi*6*x)
    dudx = u.diff(x, 1)
    points, weights = SD.points_and_weights(M,  SD.quad)
    dudx_j = np.array([dudx.subs(x, h) for h in points], dtype=np.float)
    uj = np.array([u.subs(x, h) for h in points], dtype=np.float)

    dudx_j = np.zeros(M)
    u_hat = np.zeros(M)
    u_hat = SD.forward(uj, u_hat)
    uj = SD.backward(u_hat, uj)
    u_hat = SD.forward(uj, u_hat)

    uc_hat = np.zeros(M)
    uc_hat = SD.CT.forward(uj, uc_hat)
    du_hat = np.zeros(M)
    dudx_j = SD.CT.fast_derivative(uj, dudx_j)

    Cm = CDDmat(np.arange(M).astype(np.float))
    TDMASolver = TDMA(SD)

    cs = np.zeros_like(u_hat)
    cs = Cm.matvec(u_hat, cs)

    # Should equal (but not exact so use extra resolution)
    cs2 = np.zeros(M)
    cs2 = SD.scalar_product(dudx_j, cs2)

    assert np.allclose(cs, cs2)

    cs = TDMASolver(cs)
    du = np.zeros(M)
    du = SD.backward(cs, du)

    assert np.linalg.norm(du-dudx_j)/M < 1e-10

    # Multidimensional version
    u3_hat = u_hat.repeat(4*4).reshape((M, 4, 4)) + 1j*u_hat.repeat(4*4).reshape((M, 4, 4))
    cs = np.zeros_like(u3_hat)
    cs = Cm.matvec(u3_hat, cs)
    cs2 = np.zeros((M, 4, 4), dtype=np.complex)
    du3 = dudx_j.repeat(4*4).reshape((M, 4, 4)) + 1j*dudx_j.repeat(4*4).reshape((M, 4, 4))
    cs2 = SD.scalar_product(du3, cs2)

    assert np.allclose(cs, cs2, 1e-10)

    cs = TDMASolver(cs)
    d3 = np.zeros((M, 4, 4), dtype=np.complex)
    d3 = SD.backward(cs, d3)

    #from IPython import embed; embed()
    assert np.linalg.norm(du3-d3)/(M*16) < 1e-10

#test_CDDmat('GL')

@pytest.mark.parametrize('mat', (CDDmat, CDNmat, CNDmat))
def test_CXXmat(mat):
    Cm = mat(np.arange(N).astype(np.float))
    S2 = Cm.trialfunction
    S1 = Cm.testfunction

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
    df = S2.CT.fast_derivative(fj, df)
    cs2 = np.zeros(N)
    cs2 = S1.scalar_product(df, cs2)

    assert np.allclose(cs, cs2)

    # Multidimensional version
    f_hat = f_hat.repeat(4*4).reshape((N, 4, 4)) + 1j*f_hat.repeat(4*4).reshape((N, 4, 4))
    df = df.repeat(4*4).reshape((N, 4, 4)) + 1j*df.repeat(4*4).reshape((N, 4, 4))
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)
    cs2 = np.zeros((N, 4, 4), dtype=np.complex)
    cs2 = S1.scalar_product(df, cs2)

    assert np.allclose(cs, cs2)

#test_CXXmat(CNDmat)

@pytest.mark.parametrize('quad', quads)
def test_CDTmat(quad):
    SD = cbases.ShenDirichletBasis(quad=quad)
    ST = cbases.ChebyshevBasis(quad=quad)

    Cm = CDTmat(np.arange(N).astype(np.float))

    fj = np.random.randn(N)
    # project to ST
    f_hat = np.zeros(N)
    f_hat = ST.forward(fj, f_hat)
    fj = ST.backward(f_hat, fj)

    # Check SD.scalar_product(f) equals Cm*ST.forward(f)
    f_hat = ST.forward(fj, f_hat)
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)
    df = np.zeros(N)
    df = ST.fast_derivative(fj, df)
    cs2 = np.zeros(N)
    cs2 = SD.scalar_product(df, cs2)

    #from IPython import embed; embed()
    assert np.allclose(cs, cs2)

    # Multidimensional version
    f_hat = f_hat.repeat(4*4).reshape((N, 4, 4)) + 1j*f_hat.repeat(4*4).reshape((N, 4, 4))
    df = df.repeat(4*4).reshape((N, 4, 4)) + 1j*df.repeat(4*4).reshape((N, 4, 4))
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)
    cs2 = np.zeros((N, 4, 4), dtype=np.complex)
    cs2 = SD.scalar_product(df, cs2)

    assert np.allclose(cs, cs2)

#test_CDTmat((ShenDirichletBasis('GL'), ChebyshevBasis('GL')))

@pytest.mark.parametrize('quad', quads)
def test_CTDmat(quad):
    SD = cbases.ShenDirichletBasis(quad=quad)
    ST = cbases.ChebyshevBasis(quad=quad)

    Cm = CTDmat(np.arange(N).astype(np.float))

    fj = np.random.randn(N)
    # project to SD
    f_hat = np.zeros(N)
    f_hat = SD.forward(fj, f_hat)
    fj = SD.backward(f_hat, fj)

    # Check if ST.fcs(f') equals Cm*SD.forward(f)
    f_hat = SD.forward(fj, f_hat)
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)
    df = np.zeros(N)
    df = ST.fast_derivative(fj, df)
    cs2 = np.zeros(N)
    cs2 = ST.scalar_product(df, cs2)

    #from IPython import embed; embed()
    assert np.allclose(cs, cs2)

    # Multidimensional version
    f_hat = f_hat.repeat(4*4).reshape((N, 4, 4)) + 1j*f_hat.repeat(4*4).reshape((N, 4, 4))
    df = df.repeat(4*4).reshape((N, 4, 4)) + 1j*df.repeat(4*4).reshape((N, 4, 4))
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)
    cs2 = np.zeros((N, 4, 4), dtype=np.complex)
    cs2 = ST.scalar_product(df, cs2)

    assert np.allclose(cs, cs2)

#test_CTDmat((ShenDirichletBasis('GC'), ChebyshevBasis('GC')))

@pytest.mark.parametrize('quad', quads)
def test_CDBmat(quad):
    SB = cbases.ShenBiharmonicBasis(quad=quad)
    SD = cbases.ShenDirichletBasis(quad=quad)

    M = 8*N
    Cm = CDBmat(np.arange(M).astype(np.float))

    x = Symbol("x")
    u = sin(2*pi*x)**2
    f = u.diff(x, 1)

    points, weights = SB.points_and_weights(M,  SB.quad)

    uj = np.array([u.subs(x, j) for j in points], dtype=float)
    fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points

    # project to SB
    f_hat = np.zeros(M)
    f_hat = SB.forward(uj, f_hat)
    uj = SB.backward(f_hat, uj)

    # Check if SD.scalar_product(f') equals Cm*SD.forward(f)
    f_hat = SB.forward(uj, f_hat)
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)

    df = np.zeros(M)
    df = SB.CT.fast_derivative(uj, df)
    cs2 = np.zeros(M)
    cs2 = SD.scalar_product(df, cs2)

    #from IPython import embed; embed()
    assert np.allclose(cs, cs2)

    # Multidimensional version
    f_hat = f_hat.repeat(4*4).reshape((M, 4, 4)) + 1j*f_hat.repeat(4*4).reshape((M, 4, 4))
    df = df.repeat(4*4).reshape((M, 4, 4)) + 1j*df.repeat(4*4).reshape((M, 4, 4))
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)
    cs2 = np.zeros((M, 4, 4), dtype=np.complex)
    cs2 = SD.scalar_product(df, cs2)

    assert np.allclose(cs, cs2)

#test_CDBmat((ShenBiharmonicBasis("GC"), ShenDirichletBasis("GC")))

@pytest.mark.parametrize('quad', quads)
def test_CBDmat(quad):
    SB = cbases.ShenBiharmonicBasis(quad=quad)
    SD = cbases.ShenDirichletBasis(quad=quad)

    M = 4*N
    Cm = CBDmat(np.arange(M).astype(np.float))

    x = Symbol("x")
    u = sin(12*pi*x)**2
    f = u.diff(x, 1)

    points, weights = SD.points_and_weights(M,  SD.quad)

    uj = np.array([u.subs(x, j) for j in points], dtype=float)
    fj = np.array([f.subs(x, j) for j in points], dtype=float)     # Get f on quad points

    # project to SD
    f_hat = np.zeros(M)
    f_hat = SD.forward(uj, f_hat)
    uj = SD.backward(f_hat, uj)

    # Check if SB.scalar_product(f') equals Cm*SD.forward(f)
    f_hat = SD.forward(uj, f_hat)
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)

    df = np.zeros(M)
    df = SD.CT.fast_derivative(uj, df)
    cs2 = np.zeros(M)
    cs2 = SB.scalar_product(df, cs2)

    #from IPython import embed; embed()
    assert np.allclose(cs, cs2)

    # Multidimensional version
    f_hat = f_hat.repeat(4*4).reshape((M, 4, 4)) + 1j*f_hat.repeat(4*4).reshape((M, 4, 4))
    df = df.repeat(4*4).reshape((M, 4, 4)) + 1j*df.repeat(4*4).reshape((M, 4, 4))
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)
    cs2 = np.zeros((M, 4, 4), dtype=np.complex)
    cs2 = SB.scalar_product(df, cs2)

    assert np.allclose(cs, cs2)

#test_CBDmat((ShenBiharmonicBasis("GL"), ShenDirichletBasis("GL")))


@pytest.mark.parametrize('ST', Basis[1:3])
@pytest.mark.parametrize('quad', quads)
def test_ADDmat(ST, quad):
    ST = ST(quad=quad)
    M = 2*N
    u = (1-x**2)*sin(np.pi*x)
    f = -u.diff(x, 2)

    points, weights = ST.points_and_weights(M,  quad)
    uj = np.array([u.subs(x, h) for h in points], dtype=np.float)
    fj = np.array([f.subs(x, h) for h in points], dtype=np.float)
    s = ST.slice(M)

    if ST.__class__.__name__ == "ShenDirichletBasis":
        A = ADDmat(np.arange(M).astype(np.float))
    elif ST.__class__.__name__ == "ShenNeumannBasis":
        A = ANNmat(np.arange(M).astype(np.float))
        fj -= np.dot(fj, weights)/weights.sum()
        uj -= np.dot(uj, weights)/weights.sum()

    f_hat = np.zeros(M)
    f_hat = ST.scalar_product(fj, f_hat)
    u_hat = np.zeros(M)
    u_hat[s] = solve(A.diags().toarray()[s,s], f_hat[s])

    u0 = np.zeros(M)
    u0 = ST.backward(u_hat, u0)

    #from IPython import embed; embed()
    assert np.allclose(u0, uj)

    u1 = np.zeros(M)
    u1 = ST.forward(uj, u1)
    c = np.zeros_like(u1)
    c = A.matvec(u1, c)

    assert np.allclose(c, f_hat)

#test_ADDmat(ShenNeumannBasis, "GL")

@pytest.mark.parametrize('quad', quads)
def test_SBBmat(quad):
    SB = cbases.ShenBiharmonicBasis(quad=quad)
    M = 72
    u = sin(4*pi*x)**2
    f = u.diff(x, 4)

    points, weights = SB.points_and_weights(M,  SB.quad)
    uj = np.array([u.subs(x, h) for h in points], dtype=np.float)
    fj = np.array([f.subs(x, h) for h in points], dtype=np.float)

    A = SBBmat(np.arange(M).astype(np.float))
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

    #from IPython import embed; embed()
    assert np.all(abs(c-f_hat)/c.max() < 1e-10)

    # Multidimensional
    c2 = (c.repeat(16).reshape((M, 4, 4))+1j*c.repeat(16).reshape((M, 4, 4)))
    u1 = (u1.repeat(16).reshape((M, 4, 4))+1j*u1.repeat(16).reshape((M, 4, 4)))

    c = np.zeros_like(u1)
    c = A.matvec(u1, c)

    assert np.allclose(c, c2)

#test_SBBmat(ShenBiharmonicBasis("GC"))

@pytest.mark.parametrize('quad', quads)
def test_ABBmat(quad):
    SB = cbases.ShenBiharmonicBasis(quad=quad)
    M = 6*N
    u = sin(6*pi*x)**2
    f = u.diff(x, 2)

    points, weights = SB.points_and_weights(M,  SB.quad)
    uj = np.array([u.subs(x, h) for h in points], dtype=np.float)
    fj = np.array([f.subs(x, h) for h in points], dtype=np.float)

    A = ABBmat(np.arange(M).astype(np.float))

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

    B = BBBmat(np.arange(M).astype(np.float), SB.quad)
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

