from __future__ import print_function
from time import time
import functools
from itertools import product
import pytest
import numpy as np
from mpi4py import MPI
from sympy import symbols, cos, sin, lambdify, exp
from shenfun.config import config
from shenfun.fourier import bases as fbases
from shenfun.chebyshev import bases as cbases
from shenfun.chebyshevu import bases as cubases
from shenfun.ultraspherical import bases as ubases
from shenfun.legendre import bases as lbases
from shenfun.laguerre import bases as lagbases
from shenfun.hermite import bases as hbases
from shenfun.jacobi import bases as jbases
from shenfun import Function, project, Dx, Array, FunctionSpace, TensorProductSpace, \
   VectorSpace, CompositeSpace, inner

comm = MPI.COMM_WORLD

for f in ['dct', 'dst', 'fft', 'ifft', 'rfft', 'irfft']:
    config['fftw'][f]['planner_effort'] = 'FFTW_ESTIMATE'

abstol = dict(f=5e-3, d=1e-10, g=1e-12)

def allclose(a, b):
    atol = abstol[a.dtype.char.lower()]
    return np.allclose(a, b, rtol=0, atol=atol)

def random_like(array):
    shape = array.shape
    dtype = array.dtype
    return np.random.random(shape).astype(dtype)

sizes = (12, 13)
@pytest.mark.parametrize('typecode', 'fFdD')
@pytest.mark.parametrize('dim', (2, 3, 4))
def test_transform(typecode, dim):
    s = (True,)
    if comm.Get_size() > 2 and dim > 2:
        s = (True, False)

    for slab in s:
        for shape in product(*([sizes]*dim)):
            bases = []
            for n in shape[:-1]:
                bases.append(FunctionSpace(n, 'F', dtype=typecode.upper()))
            bases.append(FunctionSpace(shape[-1], 'F', dtype=typecode))

            fft = TensorProductSpace(comm, bases, dtype=typecode, slab=slab)

            if comm.rank == 0:
                grid = [c.size for c in fft.subcomm]
                print('grid:{} shape:{} typecode:{}'
                      .format(grid, shape, typecode))

            U = random_like(fft.forward.input_array)

            F = fft.forward(U)
            V = fft.backward(F)
            assert allclose(V, U)

            # Alternative method
            fft.forward.input_array[...] = U
            fft.forward(kind={'fourier': 'vandermonde'})
            fft.backward(kind={'fourier': 'vandermonde'})
            V = fft.backward.output_array
            assert allclose(V, U)

            TT = VectorSpace(fft)
            U = Array(TT)
            V = Array(TT)
            F = Function(TT)
            U[:] = random_like(U)
            F = TT.forward(U, F)
            V = TT.backward(F, V)
            assert allclose(V, U)

            TM = CompositeSpace([fft, fft])
            U = Array(TM)
            V = Array(TM)
            F = Function(TM)
            U[:] = random_like(U)
            F = TM.forward(U, F)
            V = TM.backward(F, V)
            assert allclose(V, U)

            fftp = fft.get_dealiased(padding_factor=1.5)

            #fft.destroy()

            #padding = 1.5
            #bases = []
            #for n in shape[:-1]:
            #    bases.append(FunctionSpace(n, 'F', dtype=typecode.upper(), padding_factor=padding))
            #bases.append(FunctionSpace(shape[-1], 'F', dtype=typecode, padding_factor=padding))

            #fft = TensorProductSpace(comm, bases, dtype=typecode)

            if comm.rank == 0:
                grid = [c.size for c in fftp.subcomm]
                print('grid:{} shape:{} typecode:{}'
                      .format(grid, shape, typecode))

            U = random_like(fftp.forward.input_array)
            F = fftp.forward(U)

            Fc = F.copy()
            V = fftp.backward(F)
            F = fftp.forward(V)
            assert allclose(F, Fc)

            # Alternative method
            fftp.backward.input_array[...] = F
            fftp.backward()
            fftp.forward()
            V = fftp.forward.output_array
            assert allclose(F, V)

            fftp.destroy()
            fft.destroy()

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


cquads = ('GC', 'GL')
lquads = ('LG', 'GL')
lagquads = ('LG',)
hquads = ('HG',)
jquads = ('JG',)
uquads = ('QG',)
cuquads = ('GU', 'GC')

all_bases_and_quads = (list(product(cutrialBasis, cuquads))
                       +list(product(utrialBasis, uquads))
                       +list(product(ltrialBasis, lquads))
                       +list(product(ctrialBasis, cquads))
                       +list(product(jtrialBasis, jquads))
                       )
lag_bases_and_quads = list(product(latrialBasis, lagquads))
h_bases_and_quads = list(product(htrialBasis, hquads))

num_tests = 20
some_bases_and_quads = [all_bases_and_quads[i] for i in np.random.randint(0, len(all_bases_and_quads), num_tests)]

@pytest.mark.parametrize('typecode', 'dD')
@pytest.mark.parametrize('dim', (1, 2))
@pytest.mark.parametrize('ST,quad', some_bases_and_quads
                                    +lag_bases_and_quads
                                    +h_bases_and_quads)
def test_shentransform(typecode, dim, ST, quad):
    sizes = (16, 15)
    for shape in product(*([sizes]*dim)):
        bases = []
        for n in shape[:-1]:
            bases.append(FunctionSpace(n, 'F', dtype=typecode.upper()))
        bases.append(FunctionSpace(shape[-1], 'F', dtype=typecode))

        for axis in range(dim+1):
            ST0 = ST(sizes[-1]+4, quad=quad)
            bases.insert(axis, ST0)
            fft = TensorProductSpace(comm, bases, dtype=typecode)
            U = random_like(fft.forward.input_array)
            F = fft.forward(U)
            Fc = F.copy()
            V = fft.backward(F)
            F = fft.forward(U)
            assert allclose(F, Fc)
            bases.pop(axis)
            fft.destroy()

bases_and_quads = (list(product(ltrialBasis[:2], lquads))
                   +list(product(ctrialBasis[:2], cquads)))
                   #+list(product(jBasis[:2], jquads)))

axes = {2: {0: [0, 1, 2],
            1: [1, 0, 2],
            2: [2, 0, 1]},
        1: {0: [0, 1],
            1: [1, 0]}}

@pytest.mark.parametrize('typecode', 'dD')
@pytest.mark.parametrize('dim', (1, 2))
@pytest.mark.parametrize('ST,quad', bases_and_quads)
def test_project(typecode, dim, ST, quad):
    # Using sympy to compute an analytical solution
    x, y, z = symbols("x,y,z")
    sizes = (10, 9)

    funcs = {
        (1, 0): (cos(1*y)*sin(1*np.pi*x))*(1-x**2),
        (1, 1): (cos(1*x)*sin(1*np.pi*y))*(1-y**2),
        (2, 0): (sin(1*z)*cos(1*y)*sin(1*np.pi*x))*(1-x**2),
        (2, 1): (sin(1*z)*cos(1*x)*sin(1*np.pi*y))*(1-y**2),
        (2, 2): (sin(1*x)*cos(1*y)*sin(1*np.pi*z))*(1-z**2)
        }
    xs = {0:x, 1:y, 2:z}

    for shape in product(*([sizes]*dim)):
        bases = []
        for n in shape[:-1]:
            bases.append(FunctionSpace(n, 'F', dtype=typecode.upper()))
        bases.append(FunctionSpace(shape[-1], 'F', dtype=typecode))

        for axis in range(dim+1):
            ST0 = ST(2*shape[-1], quad=quad)
            bases.insert(axis, ST0)
            fft = TensorProductSpace(comm, bases, dtype=typecode, axes=axes[dim][axis])
            dfft = fft.get_orthogonal()
            X = fft.local_mesh(True)
            ue = funcs[(dim, axis)]
            uq = Array(fft, buffer=ue)
            uh = Function(fft)
            uh = fft.forward(uq, uh)
            due = ue.diff(xs[axis], 1)
            duq = Array(fft, buffer=due)
            dh = project(Dx(uh, axis, 1), dfft)
            uf = dh.backward()
            assert np.linalg.norm(uf-duq) < 1e-3, np.linalg.norm(uf-duq)
            for ax in (x for x in range(dim+1) if x is not axis):
                due = ue.diff(xs[axis], 1, xs[ax], 1)
                duq = Array(fft, buffer=due)
                uf = project(Dx(Dx(uh, axis, 1), ax, 1), dfft).backward()
                assert np.linalg.norm(uf-duq) < 1e-3

            bases.pop(axis)
            fft.destroy()
            dfft.destroy()

#lagbases_and_quads = list(product(lagBasis, lagquads))
@pytest.mark.parametrize('typecode', 'dD')
@pytest.mark.parametrize('dim', (1, 2))
#@pytest.mark.parametrize('ST,quad', lagbases_and_quads)
def test_project_lag(typecode, dim):
    # Using sympy to compute an analytical solution
    x, y, z = symbols("x,y,z")
    sizes = (20, 17)

    funcs = {
        (1, 0): (cos(4*y)*sin(2*x))*exp(-x),
        (1, 1): (cos(4*x)*sin(2*y))*exp(-y),
        (2, 0): (sin(3*z)*cos(4*y)*sin(2*x))*exp(-x),
        (2, 1): (sin(2*z)*cos(4*x)*sin(2*y))*exp(-y),
        (2, 2): (sin(2*x)*cos(4*y)*sin(2*z))*exp(-z)
        }
    xs = {0:x, 1:y, 2:z}

    for shape in product(*([sizes]*dim)):
        bases = []
        for n in shape[:-1]:
            bases.append(FunctionSpace(n, 'F', dtype=typecode.upper()))
        bases.append(FunctionSpace(shape[-1], 'F', dtype=typecode))

        for axis in range(dim+1):
            ST1 = latrialBasis[1](3*shape[-1])
            bases.insert(axis, ST1)
            fft = TensorProductSpace(comm, bases, dtype=typecode, axes=axes[dim][axis])
            dfft = fft.get_orthogonal()
            X = fft.local_mesh(True)
            ue = funcs[(dim, axis)]
            due = ue.diff(xs[0], 1)
            u_h = project(ue, fft)
            du_h = project(due, dfft)
            du2 = project(Dx(u_h, 0, 1), dfft)
            uf = u_h.backward()
            assert np.linalg.norm(du2-du_h) < 1e-3
            bases.pop(axis)
            fft.destroy()
            dfft.destroy()

#hbases_and_quads = list(product(hBasis, hquads))
@pytest.mark.parametrize('typecode', 'dD')
@pytest.mark.parametrize('dim', (1, 2))
#@pytest.mark.parametrize('ST,quad', hbases_and_quads)
def test_project_hermite(typecode, dim):
    # Using sympy to compute an analytical solution
    x, y, z = symbols("x,y,z")
    sizes = (10, 9)

    funcs = {
        (1, 0): (cos(4*y)*sin(2*x))*exp(-x**2/2),
        (1, 1): (cos(4*x)*sin(2*y))*exp(-y**2/2),
        (2, 0): (sin(3*z)*cos(4*y)*sin(2*x))*exp(-x**2/2),
        (2, 1): (sin(2*z)*cos(4*x)*sin(2*y))*exp(-y**2/2),
        (2, 2): (sin(2*x)*cos(4*y)*sin(2*z))*exp(-z**2/2)
        }
    xs = {0:x, 1:y, 2:z}

    for shape in product(*([sizes]*dim)):
        bases = []
        for n in shape[:-1]:
            bases.append(FunctionSpace(n, 'F', dtype=typecode.upper()))
        bases.append(FunctionSpace(shape[-1], 'F', dtype=typecode))

        for axis in range(dim+1):
            ST0 = htrialBasis[0](3*shape[-1])
            bases.insert(axis, ST0)
            fft = TensorProductSpace(comm, bases, dtype=typecode, axes=axes[dim][axis])
            X = fft.local_mesh(True)
            ue = funcs[(dim, axis)]
            due = ue.diff(xs[0], 1)
            u_h = project(ue, fft)
            du_h = project(due, fft)
            du2 = project(Dx(u_h, 0, 1), fft)

            assert np.linalg.norm(du_h-du2) < 1e-5
            bases.pop(axis)
            fft.destroy()

@pytest.mark.parametrize('quad', lquads)
def test_project_2dirichlet(quad):
    x, y = symbols("x,y")
    ue = (cos(4*y)*sin(2*x))*(1-x**2)*(1-y**2)
    sizes = (18, 17)

    D0 = lbases.ShenDirichlet(sizes[0], quad=quad)
    D1 = lbases.ShenDirichlet(sizes[1], quad=quad)
    B0 = lbases.Orthogonal(sizes[0], quad=quad)
    B1 = lbases.Orthogonal(sizes[1], quad=quad)

    DD = TensorProductSpace(comm, (D0, D1))
    BD = TensorProductSpace(comm, (B0, D1))
    DB = TensorProductSpace(comm, (D0, B1))
    BB = TensorProductSpace(comm, (B0, B1))

    X = DD.local_mesh(True)
    uh = Function(DD, buffer=ue)
    dudx_hat = project(Dx(uh, 0, 1), BD)
    dx = Function(BD, buffer=ue.diff(x, 1))
    assert np.allclose(dx, dudx_hat, 0, 1e-5)

    dudy = project(Dx(uh, 1, 1), DB).backward()
    duedy = Array(DB, buffer=ue.diff(y, 1))
    assert np.allclose(duedy, dudy, 0, 1e-5)

    us_hat = Function(BB)
    uq = uh.backward()
    us = project(uq, BB, output_array=us_hat).backward()
    assert np.allclose(us, uq, 0, 1e-5)
    dudxy = project(Dx(us_hat, 0, 1) + Dx(us_hat, 1, 1), BB).backward()
    dxy = Array(BB, buffer=ue.diff(x, 1) + ue.diff(y, 1))
    assert np.allclose(dxy, dudxy, 0, 1e-5), np.linalg.norm(dxy-dudxy)
    DD.destroy()
    BD.destroy()
    DB.destroy()
    BB.destroy()

@pytest.mark.parametrize('typecode', 'dD')
@pytest.mark.parametrize('dim', (1, 2))
@pytest.mark.parametrize('ST,quad', some_bases_and_quads)
def test_eval_tensor(typecode, dim, ST, quad):
    # Using sympy to compute an analytical solution
    # Testing for Dirichlet and regular basis
    x, y, z = symbols("x,y,z")
    sizes = (28, 27)

    funcx = (1-x**2)**3*sin(np.pi*x)
    funcy = (1-y**2)**3*sin(np.pi*y)
    funcz = (1-z**2)**3*sin(np.pi*z)

    funcs = {
        (1, 0): cos(y)*funcx,
        (1, 1): cos(x)*funcy,
        (2, 0): sin(z)*cos(y)*funcx,
        (2, 1): sin(z)*cos(x)*funcy,
        (2, 2): sin(x)*cos(y)*funcz
        }
    syms = {1: (x, y), 2:(x, y, z)}
    points = None
    if comm.Get_rank() == 0:
        points = np.random.random((dim+1, 4))
    points = comm.bcast(points)
    t_0 = 0
    t_1 = 0
    t_2 = 0
    for shape in product(*([sizes]*dim)):
        #for shape in ((64, 64),):
        bases = []
        for n in shape[:-1]:
            bases.append(FunctionSpace(n, 'F', dtype=typecode.upper()))
        bases.append(FunctionSpace(shape[-1], 'F', dtype=typecode))

        for axis in range(dim+1):
            #for axis in (0,):
            ST0 = ST(shape[-1]+6, quad=quad)
            bases.insert(axis, ST0)
            # Spectral space must be aligned in nonperiodic direction, hence axes
            fft = TensorProductSpace(comm, bases, dtype=typecode, axes=axes[dim][axis])
            #print('axes', axes[dim][axis])
            #print('bases', bases)
            #print(bases[0].axis, bases[1].axis)
            X = fft.local_mesh(True)
            ue = funcs[(dim, axis)]
            ul = lambdify(syms[dim], ue, 'numpy')
            uq = ul(*points).astype(typecode)
            u_hat = Function(fft, buffer=ue)
            t0 = time()
            result = fft.eval(points, u_hat, method=0)
            t_0 += time()-t0
            assert np.allclose(uq, result, 0, 1e-2), np.linalg.norm(uq-result)
            t0 = time()
            result = fft.eval(points, u_hat, method=1)
            t_1 += time()-t0
            assert np.allclose(uq, result, 0, 1e-2)
            t0 = time()
            result = fft.eval(points, u_hat, method=2)
            t_2 += time()-t0
            assert np.allclose(uq, result, 0, 1e-2), uq/result
            result = u_hat.eval(points)
            assert np.allclose(uq, result, 0, 1e-2)

            bases.pop(axis)
            fft.destroy()
    print('method=0', t_0)
    print('method=1', t_1)
    print('method=2', t_2)

@pytest.mark.parametrize('typecode', 'dD')
@pytest.mark.parametrize('dim', (2, 3))
def test_eval_fourier(typecode, dim):
    # Using sympy to compute an analytical solution
    x, y, z = symbols("x,y,z")
    sizes = (12, 11)

    funcs = {
        2: cos(4*x) + sin(3*y),
        3: sin(3*x) + cos(4*y) + sin(2*z)
        }
    syms = {2: (x, y), 3: (x, y, z)}
    points = None
    if comm.Get_rank() == 0:
        points = np.random.random((dim, 3))
    points = comm.bcast(points)
    t_0 = 0
    t_1 = 0
    t_2 = 0
    for shape in product(*([sizes]*dim)):
        bases = []
        for n in shape[:-1]:
            bases.append(FunctionSpace(n, 'F', dtype=typecode.upper()))

        bases.append(FunctionSpace(shape[-1], 'F', dtype=typecode))
        fft = TensorProductSpace(comm, bases, dtype=typecode)
        X = fft.local_mesh(True)
        ue = funcs[dim]
        ul = lambdify(syms[dim], ue, 'numpy')
        uq = ul(*points).astype(typecode)
        u_hat = Function(fft, buffer=ue)
        t0 = time()
        result = fft.eval(points, u_hat, method=0)
        t_0 += time() - t0
        assert allclose(uq, result), print(uq, result)
        t0 = time()
        result = fft.eval(points, u_hat, method=1)
        t_1 += time() - t0
        assert allclose(uq, result)
        t0 = time()
        result = fft.eval(points, u_hat, method=2)
        t_2 += time() - t0
        assert allclose(uq, result), print(uq, result)
        fft.destroy()

    print('method=0', t_0)
    print('method=1', t_1)
    print('method=2', t_2)

@pytest.mark.parametrize('f0,f1', product(*([('C', 'L', 'F')])*2))
def test_inner(f0, f1):
    if f0 == 'F' and f1 == 'F':
        B0 = FunctionSpace(8, f0, dtype='D', domain=(-2*np.pi, 2*np.pi))
    else:
        B0 = FunctionSpace(8, f0)
    c = Array(B0, val=1)
    d = inner(1, c)
    assert abs(d-(B0.domain[1]-B0.domain[0])) < 1e-7

    B1 = FunctionSpace(8, f1)
    T = TensorProductSpace(comm, (B0, B1))
    a0 = Array(T, val=1)
    c0 = inner(1, a0)
    L = np.array([b.domain[1]-b.domain[0] for b in (B0, B1)])
    assert abs(c0-np.prod(L)) < 1e-7
    T.destroy()

    if not (f0 == 'F' or f1 == 'F'):
        B2 = FunctionSpace(8, f1, domain=(-2, 2))
        T = TensorProductSpace(comm, (B0, B1, B2))
        a0 = Array(T, val=1)
        c0 = inner(1, a0)
        L = np.array([b.domain[1]-b.domain[0] for b in (B0, B1, B2)])
        assert abs(c0-np.prod(L)) < 1e-7
        T.destroy()

@pytest.mark.parametrize('fam', ('C', 'L', 'F', 'La', 'H'))
def test_assign(fam):
    x, y = symbols("x,y")
    for bc in (None, (0, 0), (0, 0, 0, 0)):
        dtype = 'D' if fam == 'F' else 'd'
        bc = 'periodic' if fam == 'F' else bc
        if bc in ((0, 0), (0, 0, 0, 0)) and fam in ('La', 'H'):
            continue
        tol = 1e-12 if fam in ('C', 'L', 'F') else 1e-5
        N = (10, 12)
        B0 = FunctionSpace(N[0], fam, dtype=dtype, bc=bc)
        B1 = FunctionSpace(N[1], fam, dtype=dtype, bc=bc)
        u_hat = Function(B0)
        u_hat[1:4] = 1
        ub_hat = Function(B1)
        u_hat.assign(ub_hat)
        assert abs(inner(1, u_hat)-inner(1, ub_hat)) < tol
        T = TensorProductSpace(comm, (B0, B1))
        u_hat = Function(T)
        u_hat[1:4, 1:4] = 1
        Tp = T.get_refined((2*N[0], 2*N[1]))
        ub_hat = Function(Tp)
        u_hat.assign(ub_hat)
        assert abs(inner(1, u_hat)-inner(1, ub_hat)) < tol
        VT = VectorSpace(T)
        u_hat = Function(VT)
        u_hat[:, 1:4, 1:4] = 1
        Tp = T.get_refined((2*N[0], 2*N[1]))
        VTp = VectorSpace(Tp)
        ub_hat = Function(VTp)
        u_hat.assign(ub_hat)
        assert abs(inner((1, 1), u_hat)-inner((1, 1), ub_hat)) < tol
        T.destroy()
        Tp.destroy()

def test_refine():
    assert comm.Get_size() < 7
    N = (8, 9, 10)
    F0 = FunctionSpace(8, 'F', dtype='D')
    F1 = FunctionSpace(9, 'F', dtype='D')
    F2 = FunctionSpace(10, 'F', dtype='d')
    T = TensorProductSpace(comm, (F0, F1, F2), slab=True, collapse_fourier=True)
    u_hat = Function(T)
    u = Array(T)
    u[:] = np.random.random(u.shape)
    u_hat = u.forward(u_hat)
    Tp = T.get_dealiased(padding_factor=(2, 2, 2))
    u_ = Array(Tp)
    up_hat = Function(Tp)
    assert up_hat.commsizes == u_hat.commsizes
    u2 = u_hat.refine(2*np.array(N))
    V = VectorSpace(T)
    u_hat = Function(V)
    u = Array(V)
    u[:] = np.random.random(u.shape)
    u_hat = u.forward(u_hat)
    Vp = V.get_dealiased(padding_factor=(2, 2, 2))
    u_ = Array(Vp)
    up_hat = Function(Vp)
    assert up_hat.commsizes == u_hat.commsizes
    u3 = u_hat.refine(2*np.array(N))
    T.destroy()
    Tp.destroy()

def test_eval_expression():
    import sympy as sp
    from shenfun import div, grad
    x, y, z = sp.symbols('x,y,z')
    B0 = FunctionSpace(16, 'C')
    B1 = FunctionSpace(17, 'C')
    B2 = FunctionSpace(20, 'F', dtype='d')

    TB = TensorProductSpace(comm, (B0, B1, B2))
    f = sp.sin(x)+sp.sin(y)+sp.sin(z)
    dfx = f.diff(x, 2) + f.diff(y, 2) + f.diff(z, 2)
    fa = Function(TB, buffer=f)

    dfe = div(grad(fa))
    dfa = project(dfe, TB)

    xyz = np.array([[0.25, 0.5, 0.75],
                    [0.25, 0.5, 0.75],
                    [0.25, 0.5, 0.75]])

    f0 = lambdify((x, y, z), dfx)(*xyz)
    f1 = dfe.eval(xyz)
    f2 = dfa.eval(xyz)
    assert np.allclose(f0, f1, 1e-7)
    assert np.allclose(f1, f2, 1e-7)
    TB.destroy()

@pytest.mark.parametrize('fam', ('C', 'L'))
def test_eval_expression2(fam):
    import sympy as sp
    from shenfun import div, grad
    x, y = sp.symbols('x,y')
    B0 = FunctionSpace(16, fam, domain=(-2, 2))
    B1 = FunctionSpace(17, fam, domain=(-1, 3))

    T = TensorProductSpace(comm, (B0, B1))
    f = sp.sin(x)+sp.sin(y)
    dfx = f.diff(x, 2) + f.diff(y, 2)
    fa = Function(T, buffer=f)

    dfe = div(grad(fa))
    dfa = project(dfe, T)

    xy = np.array([[0.25, 0.5, 0.75],
                   [0.25, 0.5, 0.75]])

    f0 = lambdify((x, y), dfx)(*xy)
    f1 = dfe.eval(xy)
    f2 = dfa.eval(xy)
    assert np.allclose(f0, f1, 1e-7)
    assert np.allclose(f1, f2, 1e-7)
    T.destroy()


if __name__ == '__main__':
    test_transform('F', 2)
    #test_transform('d', 2)
    #test_shentransform('d', 2, jbases.Phi4, 'JG')
    #test_eval_expression()
    #test_eval_expression2('L')
    test_project('d', 2, ltrialBasis[0], 'GL')
    #test_project_lag('d', 2)
    #test_project_hermite('d', 2)
    #test_project2('d', 2, lbases.ShenNeumann, 'LG')
    #test_project_2dirichlet('GL')
    #test_eval_tensor('d', 2, cbases.ShenDirichlet, 'GC')
    #test_eval_fourier('D', 3)
    #test_inner('C', 'F')
    #test_refine()
    #test_assign('C')
