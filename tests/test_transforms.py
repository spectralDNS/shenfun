from itertools import product
import pytest
from scipy.linalg import solve
import scipy.sparse.linalg as la
from sympy import symbols, sin, cos, pi, lambdify
import numpy as np
import shenfun
from shenfun.chebyshev import bases as cbases
from shenfun.chebyshevu import bases as cubases
from shenfun.legendre import bases as lbases
from shenfun.laguerre import bases as lagbases
from shenfun.ultraspherical import bases as ubases
from shenfun.hermite import bases as hbases
from shenfun.fourier import bases as fbases
from shenfun.jacobi import bases as jbases
from shenfun.la import TDMA
from shenfun.spectralbase import inner_product
from shenfun.config import config

for f in ['dct', 'dst', 'fft', 'ifft', 'rfft', 'irfft']:
    config['fftw'][f]['planner_effort'] = 'FFTW_ESTIMATE'

N = 33
x, y = symbols("x,y", real=True)

ctrialBasis = [cbases.__dict__.get(base) for base in cbases.bases[:-1]]
ctestBasis = ctrialBasis + [cbases.__dict__.get(base) for base in cbases.testbases]
cutrialBasis = [cubases.__dict__.get(base) for base in cubases.bases[:-1]]
cutestBasis = cutrialBasis + [cubases.__dict__.get(base) for base in cubases.testbases]

utrialBasis = [ubases.__dict__.get(base) for base in ubases.bases[:-1]]
utestBasis = utrialBasis + [ubases.__dict__.get(base) for base in ubases.testbases]
ltrialBasis = [lbases.__dict__.get(base) for base in lbases.bases[:-1]]
ltestBasis = ltrialBasis + [lbases.__dict__.get(base) for base in lbases.testbases]
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


fBasis = (fbases.R2C,
          fbases.C2C)


all_trial_bases_and_quads = (list(product(latrialBasis, lagquads))
                     +list(product(ltrialBasis, lquads))
                     +list(product(ctrialBasis, cquads))
                     +list(product(utrialBasis, uquads))
                     +list(product(fBasis, ('',)))
                     +list(product(jtrialBasis, jquads)))

cbases2 = list(list(i[0]) + [i[1]] for i in product(list(product(ctestBasis, ctrialBasis)), cquads))
lbases2 = list(list(i[0]) + [i[1]] for i in product(list(product(ltestBasis, ltrialBasis)), lquads))

cl_nonortho = (list(product(latrialBasis[1:], lagquads))
               +list(product(ltrialBasis[1:], lquads))
               +list(product(ctrialBasis[1:], cquads)))

class ABC(object):
    def __init__(self, dim, coors):
        self.dim = dim
        self.coors = coors
        self.sg = coors.sg
    @property
    def dimensions(self):
        return self.dim


@pytest.mark.parametrize('basis', fBasis)
@pytest.mark.parametrize('N', (8, 9))
def test_convolve(basis, N):
    """Test convolution"""
    FFT = basis(N)
    u0 = shenfun.Function(FFT)
    u1 = shenfun.Function(FFT)
    M = u0.shape[0]
    u0[:] = np.random.rand(M) + 1j*np.random.rand(M)
    u1[:] = np.random.rand(M) + 1j*np.random.rand(M)
    if isinstance(FFT, fbases.R2C):
        # Make sure spectral data corresponds to real input
        u0[0] = u0[0].real
        u1[0] = u1[0].real
        if N % 2 == 0:
            u0[-1] = u0[-1].real
            u1[-1] = u1[-1].real
    uv1 = FFT.convolve(u0, u1, fast=False)

    # Do convolution with FFT and padding
    FFT2 = basis(N, padding_factor=(1.5+1.00001/N)) # Just enough to be perfect
    uv2 = FFT2.convolve(u0, u1, fast=True)

    # Compare. Should be identical after truncation if no aliasing
    uv3 = np.zeros_like(uv2)
    if isinstance(FFT, fbases.R2C):
        uv3[:] = uv1[:N//2+1]
        if N % 2 == 0:
            uv3[-1] *= 2
            uv3[-1] = uv3[-1].real
    else:
        uv3[:N//2+1] = uv1[:N//2+1]
        uv3[-(N//2):] += uv1[-(N//2):]
    assert np.allclose(uv3, uv2)


@pytest.mark.parametrize('ST,quad', list(product(ctrialBasis, cquads))+ list(product(ltrialBasis, ["LG"])) + list(product(fBasis, [""])))
def test_scalarproduct(ST, quad):
    """Test fast scalar product against Vandermonde computed version"""
    kwargs = {}
    if not ST.family() == 'fourier':
        kwargs['quad'] = quad
    ST = ST(N, **kwargs)
    f = x*x+cos(pi*x)
    fj = shenfun.Array(ST, buffer=f)
    u0 = shenfun.Function(ST)
    u1 = shenfun.Function(ST)
    u0 = ST.scalar_product(fj, u0, kind='fast')
    u1 = ST.scalar_product(fj, u1, kind='vandermonde')
    assert np.allclose(u1, u0)
    assert not np.all(u1 == u0) # Check that fast is not the same as slow

@pytest.mark.parametrize('ST,quad', all_trial_bases_and_quads)
def test_eval(ST, quad):
    """Test eval against fast inverse"""
    kwargs = {}
    if not ST.family() == 'fourier':
        kwargs['quad'] = quad
    ST = ST(N, **kwargs)
    points, weights = ST.points_and_weights(N)
    fk = shenfun.Function(ST)
    fk[:4] = 1
    ST.eval(np.array([0.]), fk)
    f = ST.eval(points, fk)
    fj = fk.backward()
    assert np.allclose(fj, f, rtol=1e-5, atol=1e-6), np.linalg.norm(fj-f)
    fj = ST.backward(fk, fj, kind='vandermonde')
    fk = ST.forward(fj, fk, kind='vandermonde')
    f = ST.eval(points, fk)
    assert np.allclose(fj, f, rtol=1e-5, atol=1e-6), np.linalg.norm(fj-f)

@pytest.mark.parametrize('basis, quad', cl_nonortho)
#@pytest.mark.xfail(raises=AssertionError)
def test_to_ortho(basis, quad):
    N = 10
    if basis.family() == 'legendre':
        B1 = ltrialBasis[0](N, quad)
        #B3 = lBasis[0](N, quad)
    elif basis.family() == 'chebyshev':
        B1 = ctrialBasis[0](N, quad)
        #B3 = cBasis[0](N, quad)
    elif basis.family() == 'laguerre':
        B1 = latrialBasis[0](N)
        #B3 = laBasis[0](N)

    B0 = basis(N, quad=quad)
    a = shenfun.Array(B0)
    a_hat = shenfun.Function(B0)
    b0_hat = shenfun.Function(B1)
    b1_hat = shenfun.Function(B1)
    a[:] = np.random.random(a.shape)
    a_hat = a.forward(a_hat)
    b0_hat = shenfun.project(a_hat, B1, output_array=b0_hat, fill=False,  use_to_ortho=True)
    b1_hat = shenfun.project(a_hat, B1, output_array=b1_hat, fill=False,  use_to_ortho=False)
    assert np.linalg.norm(b0_hat-b1_hat) < 1e-10

    #B2 = basis(N, quad=quad)
    TD = shenfun.TensorProductSpace(shenfun.comm, (B0, B0))
    TC = shenfun.TensorProductSpace(shenfun.comm, (B1, B1))
    a = shenfun.Array(TD)
    a_hat = shenfun.Function(TD)
    b0_hat = shenfun.Function(TC)
    b1_hat = shenfun.Function(TC)
    a[:] = np.random.random(a.shape)
    a_hat = a.forward(a_hat)
    b0_hat = shenfun.project(a_hat, TC, output_array=b0_hat, fill=False, use_to_ortho=True)
    b1_hat = shenfun.project(a_hat, TC, output_array=b1_hat, fill=False, use_to_ortho=False)
    assert np.linalg.norm(b0_hat-b1_hat) < 1e-10
    TD.destroy()
    TC.destroy()

    F0 = shenfun.FunctionSpace(N, 'F')
    TD = shenfun.TensorProductSpace(shenfun.comm, (B0, F0))
    TC = shenfun.TensorProductSpace(shenfun.comm, (B1, F0))
    a = shenfun.Array(TD)
    a_hat = shenfun.Function(TD)
    b0_hat = shenfun.Function(TC)
    b1_hat = shenfun.Function(TC)
    a[:] = np.random.random(a.shape)
    a_hat = a.forward(a_hat)
    b0_hat = shenfun.project(a_hat, TC, output_array=b0_hat, fill=False, use_to_ortho=True)
    b1_hat = shenfun.project(a_hat, TC, output_array=b1_hat, fill=False, use_to_ortho=False)
    assert np.linalg.norm(b0_hat-b1_hat) < 1e-10
    TD.destroy()
    TC.destroy()


@pytest.mark.parametrize('test, trial, quad', cbases2+lbases2)
def test_massmatrices(test, trial, quad):
    test = test(N, quad=quad)
    trial = trial(N, quad=quad)
    f_hat = np.zeros(N)
    fj = shenfun.Array(trial)
    fj[:trial.dim()] = np.random.random(trial.dim())
    f_hat = trial.forward(fj, f_hat)
    fj = trial.backward(f_hat, fj)
    BBD = inner_product((test, 0), (trial, 0)) #, assemble='quadrature_vandermonde')
    f_hat = trial.forward(fj, f_hat)
    u2 = np.zeros_like(f_hat)
    u2 = BBD.matvec(f_hat, u2)
    u0 = np.zeros(N)
    u0 = test.scalar_product(fj, u0)
    s = test.slice()
    assert np.allclose(u0[s], u2[s], rtol=1e-5, atol=1e-6)
    del BBD

@pytest.mark.parametrize('basis', ctrialBasis[:2])
def test_project_1D(basis):
    ue = sin(2*np.pi*x)*(1-x**2)
    T = basis(12)
    u = shenfun.TrialFunction(T)
    v = shenfun.TestFunction(T)
    u_tilde = shenfun.Function(T)
    ua = shenfun.Array(T, buffer=ue)
    u_tilde = shenfun.inner(v, ua, output_array=u_tilde)
    M = shenfun.inner(u, v)
    u_p = shenfun.Function(T)
    u_p = M.solve(u_tilde, u=u_p)
    u_0 = shenfun.Function(T)
    u_0 = shenfun.project(ua, T)
    assert np.allclose(u_0, u_p)
    u_1 = shenfun.project(ue, T)
    assert np.allclose(u_1, u_p)

@pytest.mark.parametrize('ST,quad', all_trial_bases_and_quads)
def test_transforms(ST, quad):
    N = 10
    kwargs = {}
    if not ST.family() == 'fourier':
        kwargs['quad'] = quad
    ST0 = ST(N, **kwargs)
    fj = shenfun.Array(ST0)
    fj[:] = np.random.random(N)
    fj = fj.forward().backward().copy()
    assert np.allclose(fj, fj.forward().backward())
    u0 = shenfun.Function(ST0)
    u1 = shenfun.Array(ST0)
    u0 = ST0.forward(fj, u0, kind='vandermonde')
    u1 = ST0.backward(u0, u1, kind='vandermonde')
    assert np.allclose(fj, u1, rtol=1e-5, atol=1e-6)

    # Multidimensional version
    ST0 = ST(N, **kwargs)
    if ST0.short_name() in ('R2C', 'C2C'):
        F0 = shenfun.FunctionSpace(N, 'F', dtype='D')
        T0 = shenfun.TensorProductSpace(shenfun.comm, (F0, ST0))

    else:
        F0 = shenfun.FunctionSpace(N, 'F', dtype='d')
        T0 = shenfun.TensorProductSpace(shenfun.comm, (F0, ST0))
    fij = shenfun.Array(T0)
    fij[:] = np.random.random(T0.shape(False))
    fij = fij.forward().backward().copy()
    assert np.allclose(fij, fij.forward().backward())
    T0.destroy()

    if ST0.short_name() in ('R2C', 'C2C'):
        F0 = shenfun.FunctionSpace(N, 'F', dtype='D')
        F1 = shenfun.FunctionSpace(N, 'F', dtype='D')
        T = shenfun.TensorProductSpace(shenfun.comm, (F0, F1, ST0), dtype=ST0.dtype.char)

    else:
        F0 = shenfun.FunctionSpace(N, 'F', dtype='d')
        F1 = shenfun.FunctionSpace(N, ST.family())
        T = shenfun.TensorProductSpace(shenfun.comm, (F0, ST0, F1))

    fij = shenfun.Array(T)
    fij[:] = np.random.random(T.shape(False))
    fij = fij.forward().backward().copy()
    assert np.allclose(fij, fij.forward().backward())
    T.destroy()


@pytest.mark.parametrize('ST,quad', all_trial_bases_and_quads)
@pytest.mark.parametrize('axis', (0, 1, 2))
def test_axis(ST, quad, axis):
    kwargs = {}
    if not ST.family() == 'fourier':
        kwargs['quad'] = quad
    ST = ST(N, **kwargs)
    points, weights = ST.points_and_weights(N)
    f_hat = shenfun.Function(ST)
    f_hat[ST.slice()] = np.random.random(f_hat[ST.slice()].shape)

    B = inner_product((ST, 0), (ST, 0))
    c = shenfun.Function(ST)
    c = B.solve(f_hat, c)

    # Multidimensional version
    #f0 = shenfun.Array(ST)
    #bc = [np.newaxis,]*3
    #bc[axis] = slice(None)
    #ST.tensorproductspace = ABC(3, ST.coors)
    #ST.plan((N,)*3, axis, f0.dtype, {})
    #if ST.has_nonhomogeneous_bcs:
    #    ST.bc.set_tensor_bcs(ST, ST) # To set Dirichlet boundary conditions on multidimensional array
    #ck = shenfun.Function(ST)
    #fk = np.broadcast_to(f_hat[tuple(bc)], ck.shape).copy()
    #ck = B.solve(fk, ck, axis=axis)
    #cc = [1,]*3
    #cc[axis] = slice(None)
    #assert np.allclose(ck[tuple(cc)], c, rtol=1e-5, atol=1e-6)

@pytest.mark.parametrize('quad', cquads)
def test_CDDmat(quad):
    M = 48
    SD = cbases.ShenDirichlet(M, quad=quad)
    u = (1-x**2)*sin(np.pi*6*x)
    dudx = u.diff(x, 1)
    dudx_hat = shenfun.Function(SD, buffer=dudx)
    u_hat = shenfun.Function(SD, buffer=u)
    ducdx_hat = shenfun.project(shenfun.Dx(u_hat, 0, 1), SD)
    assert np.linalg.norm(ducdx_hat-dudx_hat)/M < 1e-10, np.linalg.norm(ducdx_hat-dudx_hat)/M

    # Multidimensional version
    SD0 = cbases.ShenDirichlet(8, quad=quad)
    SD1 = cbases.ShenDirichlet(M, quad=quad)
    T = shenfun.TensorProductSpace(shenfun.comm, (SD0, SD1))
    u = (1-y**2)*sin(np.pi*6*y)
    dudy = u.diff(y, 1)
    dudy_hat = shenfun.Function(T, buffer=dudy)
    u_hat = shenfun.Function(T, buffer=u)
    ducdy_hat = shenfun.project(shenfun.Dx(u_hat, 1, 1), T)
    assert np.linalg.norm(ducdy_hat-dudy_hat)/M < 1e-10, np.linalg.norm(ducdy_hat-dudy_hat)/M
    T.destroy()

@pytest.mark.parametrize('test,trial', product(ctestBasis, ctrialBasis))
def test_CXXmat(test, trial):
    test = test(N)
    trial = trial(N)

    CT = ctrialBasis[0](N)

    Cm = inner_product((test, 0), (trial, 1))
    S2 = Cm.trialfunction[0]
    S1 = Cm.testfunction[0]

    #initialize random vector
    f_hat = shenfun.Function(S2)
    f_hat[:S2.dim()] = np.random.rand(S2.dim())
    fj = f_hat.backward()

    # Check S1.scalar_product(f) equals Cm*S2.forward(f)
    f_hat = S2.forward(fj, f_hat)
    cs = np.zeros_like(f_hat)
    cs = Cm.matvec(f_hat, cs)
    df = shenfun.project(shenfun.grad(f_hat), CT).backward()
    cs2 = np.zeros(N)
    cs2 = S1.scalar_product(df, cs2)
    s = S1.slice()
    assert np.allclose(cs[s], cs2[s], rtol=1e-5, atol=1e-6)


dirichlet_with_quads = (list(product([cbases.ShenNeumann, cbases.ShenDirichlet], cquads)) +
                        list(product([lbases.ShenNeumann, lbases.ShenDirichlet], lquads)))

@pytest.mark.parametrize('ST,quad', dirichlet_with_quads)
def test_ASDSDmat(ST, quad):
    M = 2*N
    ST = ST(M, quad=quad)
    u = (1-x**2)*sin(np.pi*x)
    f = u.diff(x, 2)
    ul = lambdify(x, u, 'numpy')
    fl = lambdify(x, f, 'numpy')
    points, weights = ST.points_and_weights(M)
    uj = ul(points)
    fj = fl(points)
    s = ST.slice()

    if ST.family() == 'chebyshev':
        A = inner_product((ST, 0), (ST, 2))
    else:
        A = inner_product((ST, 1), (ST, 1))

    f_hat = np.zeros(M)
    f_hat = ST.scalar_product(fj, f_hat)
    if ST.family() == 'legendre':
        f_hat *= -1

    # Test both solve interfaces
    c_hat = f_hat.copy()
    constraints = ((0, 0),) if A.__class__.__name__ == 'ASNSNmat' else ()
    c_hat = A.solve(c_hat, constraints=constraints)

    u_hat = np.zeros_like(f_hat)
    u_hat = A.solve(f_hat, u_hat, constraints=constraints)

    assert np.allclose(c_hat[s], u_hat[s], rtol=1e-5, atol=1e-6)

    u0 = np.zeros(M)
    u0 = ST.backward(u_hat, u0)
    assert np.allclose(u0, uj)

    u1 = np.zeros(M)
    u1 = ST.forward(uj, u1)
    c = np.zeros_like(u1)
    c = A.matvec(u1, c)
    s = ST.slice()
    assert np.allclose(c[s], f_hat[s], rtol=1e-5, atol=1e-6)

    # Multidimensional
    c_hat = f_hat.copy()
    c_hat = c_hat.repeat(M).reshape((M, M)).transpose().copy()
    c_hat = A.solve(c_hat, axis=1, constraints=constraints)
    assert np.allclose(c_hat[0, s], u_hat[s], rtol=1e-5, atol=1e-6)

biharmonic_with_quads = (list(product([cbases.ShenBiharmonic], cquads)) +
                         list(product([lbases.ShenBiharmonic], lquads)))

@pytest.mark.parametrize('SB,quad', biharmonic_with_quads)
def test_SSBSBmat(SB, quad):
    M = 72
    SB = SB(M, quad=quad)
    u = sin(4*pi*x)**2
    f = u.diff(x, 4)
    ul = lambdify(x, u, 'numpy')
    fl = lambdify(x, f, 'numpy')
    points, weights = SB.points_and_weights(M)
    uj = ul(points)
    fj = fl(points)

    if SB.family() == 'chebyshev':
        A = inner_product((SB, 0), (SB, 4))
    else:
        A = inner_product((SB, 2), (SB, 2))
    f_hat = np.zeros(M)
    f_hat = SB.scalar_product(fj, f_hat)
    u_hat = np.zeros(M)
    u_hat = A.solve(f_hat, u_hat)

    u0 = np.zeros(M)
    u0 = SB.backward(u_hat, u0)

    assert np.allclose(u0, uj, rtol=1e-5, atol=1e-6)

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

@pytest.mark.parametrize('SB,quad', biharmonic_with_quads)
def test_ASBSBmat(SB, quad):
    M = 4*N
    SB = SB(M, quad=quad)
    u = sin(6*pi*x)**2
    f = u.diff(x, 2)
    fl = lambdify(x, f, "numpy")
    ul = lambdify(x, u, "numpy")

    points, weights = SB.points_and_weights(M)
    uj = ul(points)
    fj = fl(points)

    f_hat = np.zeros(M)
    f_hat = SB.scalar_product(fj, f_hat)
    if SB.family() == 'chebyshev':
        A = inner_product((SB, 0), (SB, 2))
    else:
        A = inner_product((SB, 1), (SB, 1))
        f_hat *= -1.0

    u_hat = np.zeros(M)
    u_hat = A.solve(f_hat, u_hat)

    u0 = np.zeros(M)
    u0 = SB.backward(u_hat, u0)
    assert np.allclose(u0, uj), np.linalg.norm(u0-uj)

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

    B = inner_product((SB, 0), (SB, 0))

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
    assert np.allclose(z0, u0, rtol=1e-5, atol=1e-6)

def test_dlt():
    for N in (10, 500, 1000):
        L = shenfun.FunctionSpace(N, 'L')
        T = shenfun.FunctionSpace(N, 'C')
        cleg = shenfun.Function(L, val=1)
        cc = shenfun.Function(T)
        C = shenfun.legendre.dlt.Leg2Cheb(cleg, use_direct=100)
        A = shenfun.legendre.dlt.Cheb2Leg(cc, use_direct=100)
        assert np.linalg.norm(A(C(cleg))-cleg) < 1e-8
        cc = C(cleg, cc, transpose=False)
        x = np.array([0.5, 0.6])
        assert np.linalg.norm(cc(x)-cleg(x)) < 1e-10, np.linalg.norm(cc(x)-cleg(x))
        e0 = np.linalg.norm(cleg - cleg.backward(kind='recursive').forward(kind='recursive'))
        assert e0 < 1e-8, e0
        e1 = np.linalg.norm(cleg - cleg.backward(kind='fast').forward(kind='fast'))
        assert e1 < 1e-8, e1
        cc = C(cleg, cc, transpose=True)
        e2 = np.linalg.norm(cc-1)
        assert e2 < 1e-8, e2
    N = 500
    L = shenfun.FunctionSpace(N, 'L')
    T = shenfun.FunctionSpace(N, 'C')
    F = shenfun.FunctionSpace(6, 'F', dtype='d')
    F1 = shenfun.FunctionSpace(6, 'F', dtype='D')
    TL = shenfun.TensorProductSpace(shenfun.comm, (F, L), dtype='d')
    TT = shenfun.TensorProductSpace(shenfun.comm, (F, T), dtype='d')
    cl = shenfun.Function(TL, val=1)
    cb1 = cl.backward(kind={'legendre': 'recursive'})
    cb2 = cl.backward(kind={'legendre': 'fast'})
    cb3 = cl.backward(kind={'legendre': 'vandermonde'})
    assert np.linalg.norm(cb1-cb2) < 1e-8
    assert np.linalg.norm(cb1-cb3) < 1e-8
    CC = shenfun.legendre.dlt.Leg2chebHaleTownsend(cl, axis=1)
    c2 = shenfun.Function(TT)
    c2 = CC(cl, c2)
    xx = np.random.rand(2, 4)
    assert np.linalg.norm(c2(xx)-cl(xx)) < 1e-8
    c2 = CC(cl, c2, transpose=True)
    assert np.linalg.norm(c2-1) < 1e-8
    c3 = np.zeros_like(c2)
    c3 = shenfun.legendre.dlt.leg2cheb(cl, c3, axis=1, transpose=True)
    assert np.linalg.norm(c3-1) < 1e-8
    c3 = shenfun.legendre.dlt.leg2cheb(cl, c3, axis=1, transpose=False)
    assert np.linalg.norm(c3(xx)-cl(xx)) < 1e-8
    TL = shenfun.TensorProductSpace(shenfun.comm, (F1, L, F), dtype='d')
    TT = shenfun.TensorProductSpace(shenfun.comm, (F1, T, F), dtype='d')
    cl = shenfun.Function(TL, val=1)
    CC = shenfun.legendre.dlt.Leg2chebHaleTownsend(cl, axis=1)
    c2 = shenfun.Function(TT)
    c2 = CC(cl, c2)
    xx = np.random.rand(3, 4)
    assert np.linalg.norm(c2(xx)-cl(xx)) < 1e-8
    assert np.linalg.norm(cl - cl.backward(kind={'legendre': 'recursive'}).forward(kind={'legendre': 'recursive'})) < 1e-8
    assert np.linalg.norm(cl - cl.backward(kind={'legendre': 'fast'}).forward(kind={'legendre': 'fast'})) < 1e-8
    cb1 = cl.backward(kind={'legendre': 'recursive'})
    cb2 = cl.backward(kind={'legendre': 'fast'})
    assert np.linalg.norm(cb1-cb2) < 1e-7, np.linalg.norm(cb1-cb2)

def test_leg2cheb():
    for N in (100, 701, 1200):
        #u = np.random.random(N)
        u = np.ones(N)
        C = shenfun.legendre.dlt.Leg2Cheb(u, use_direct=100)
        A = shenfun.legendre.dlt.Cheb2Leg(u, use_direct=100)
        assert np.linalg.norm(A(C(u))-u) < 1e-8
        for level in range(1, 4):
            C = shenfun.legendre.dlt.Leg2Cheb(u, levels=level, use_direct=100)
            A = shenfun.legendre.dlt.Cheb2Leg(u, levels=level, use_direct=100)
            assert np.linalg.norm(A(C(u))-u) < 1e-8
        for domains in range(3, 6):
            C = shenfun.legendre.dlt.Leg2Cheb(u, domains=domains, use_direct=100)
            A = shenfun.legendre.dlt.Cheb2Leg(u, domains=domains, use_direct=100)
            assert np.linalg.norm(A(C(u))-u) < 1e-8
        C2 = shenfun.legendre.dlt.Leg2chebHaleTownsend(u)
        assert np.linalg.norm(C2(u, transpose=True)-1) < 1e-8
        assert np.linalg.norm(C2(u)-C(u)) < 1e-8

if __name__ == '__main__':
    from time import time
    config['optimization']['mode'] = 'cython'
    #test_to_ortho(cBasisGC[1], 'GC')
    # test_convolve(fbases.R2C, 8)
    #test_ASDSDmat(cbases.ShenNeumann, "GC")
    #test_ASBSBmat(cbases.ShenBiharmonic, "GC")
    #test_CDDmat("GL")
    #for i in range(len(ctrialBasis)):
    #    test_massmatrices(ctestBasis[-4], ctrialBasis[i], 'GL')
    #test_massmatrices(ctestBasis[-4], ctrialBasis[2], 'GL')
    #test_CXXmat(ctestBasis[3], ctrialBasis[1])
    #test_transforms(cBasisGC[3], 'GC')
    #test_project_1D(cBasis[0])
    #test_scalarproduct(ltrialBasis[2], 'LG')
    test_dlt()
    #test_leg2cheb()
    #test_eval(cuBasis[-1], 'GU')
    #test_axis(laBasis[1], 'LG', 1)
