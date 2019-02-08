from __future__ import print_function
from time import time
from itertools import product
import pytest
import numpy as np
from mpi4py import MPI
from sympy import symbols, cos, sin, lambdify, exp
from shenfun.chebyshev import bases as cbases
from shenfun.legendre import bases as lbases
from shenfun.laguerre import bases as lagbases
from shenfun.hermite import bases as hbases
from shenfun import Function, project, Dx, Array, Basis, TensorProductSpace, \
   VectorTensorProductSpace, MixedTensorProductSpace

comm = MPI.COMM_WORLD

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
                bases.append(Basis(n, 'F', dtype=typecode.upper()))
            bases.append(Basis(shape[-1], 'F', dtype=typecode))

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
            fft.forward(fast_transform=False)
            fft.backward(fast_transform=False)
            V = fft.backward.output_array
            assert allclose(V, U)

            TT = VectorTensorProductSpace(fft)
            U = Array(TT)
            V = Array(TT)
            F = Function(TT)
            U[:] = random_like(U)
            F = TT.forward(U, F)
            V = TT.backward(F, V)
            assert allclose(V, U)

            TM = MixedTensorProductSpace([fft, fft])
            U = Array(TM)
            V = Array(TM)
            F = Function(TM)
            U[:] = random_like(U)
            F = TM.forward(U, F)
            V = TM.backward(F, V)
            assert allclose(V, U)

            fft.destroy()

            padding = 1.5
            bases = []
            for n in shape[:-1]:
                bases.append(Basis(n, 'F', dtype=typecode.upper(), padding_factor=padding))
            bases.append(Basis(shape[-1], 'F', dtype=typecode, padding_factor=padding))

            fft = TensorProductSpace(comm, bases, dtype=typecode)

            if comm.rank == 0:
                grid = [c.size for c in fft.subcomm]
                print('grid:{} shape:{} typecode:{}'
                      .format(grid, shape, typecode))

            U = random_like(fft.forward.input_array)
            F = fft.forward(U)

            Fc = F.copy()
            V = fft.backward(F)
            F = fft.forward(V)
            assert allclose(F, Fc)

            # Alternative method
            fft.backward.input_array[...] = F
            fft.backward()
            fft.forward()
            V = fft.forward.output_array
            assert allclose(F, V)

            fft.destroy()

cBasis = (cbases.Basis,
          cbases.ShenDirichletBasis,
          cbases.ShenNeumannBasis,
          cbases.ShenBiharmonicBasis)

lBasis = (lbases.Basis,
          lbases.ShenDirichletBasis,
          lbases.ShenNeumannBasis,
          lbases.ShenBiharmonicBasis)

lagBasis = (lagbases.Basis,
            lagbases.ShenDirichletBasis)

hBasis = (hbases.Basis,)

cquads = ('GC', 'GL')
lquads = ('LG', 'GL')
lagquads = ('LG',)
hquads = ('HG',)

all_bases_and_quads = list(product(lBasis, lquads))+list(product(cBasis, cquads))
lag_bases_and_quads = list(product(lagBasis, lagquads))
h_bases_and_quads = list(product(hBasis, hquads))

@pytest.mark.parametrize('typecode', 'dD')
@pytest.mark.parametrize('dim', (1, 2))
@pytest.mark.parametrize('ST,quad', all_bases_and_quads+lag_bases_and_quads+h_bases_and_quads)
def test_shentransform(typecode, dim, ST, quad):
    for shape in product(*([sizes]*dim)):
        bases = []
        for n in shape[:-1]:
            bases.append(Basis(n, 'F', dtype=typecode.upper()))
        bases.append(Basis(shape[-1], 'F', dtype=typecode))

        for axis in range(dim+1):
            ST0 = ST(shape[-1], quad=quad)
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
#test_shentransform('d', 2, lBasis[0], 'LG')

bases_and_quads = list(product(lBasis[:2], lquads))+list(product(cBasis[:2], cquads))

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
    sizes = (24, 23)

    funcs = {
        (1, 0): (cos(4*y)*sin(2*np.pi*x))*(1-x**2),
        (1, 1): (cos(4*x)*sin(2*np.pi*y))*(1-y**2),
        (2, 0): (sin(6*z)*cos(4*y)*sin(2*np.pi*x))*(1-x**2),
        (2, 1): (sin(2*z)*cos(4*x)*sin(2*np.pi*y))*(1-y**2),
        (2, 2): (sin(2*x)*cos(4*y)*sin(2*np.pi*z))*(1-z**2)
        }
    syms = {1: (x, y), 2:(x, y, z)}
    xs = {0:x, 1:y, 2:z}

    for shape in product(*([sizes]*dim)):
        bases = []
        for n in shape[:-1]:
            bases.append(Basis(n, 'F', dtype=typecode.upper()))
        bases.append(Basis(shape[-1], 'F', dtype=typecode))

        for axis in range(dim+1):
            ST0 = ST(shape[-1], quad=quad)
            bases.insert(axis, ST0)
            fft = TensorProductSpace(comm, bases, dtype=typecode, axes=axes[dim][axis])
            X = fft.local_mesh(True)
            ue = funcs[(dim, axis)]
            ul = lambdify(syms[dim], ue, 'numpy')
            uq = ul(*X).astype(typecode)
            uh = Function(fft)
            uh = fft.forward(uq, uh)
            due = ue.diff(xs[axis], 1)
            dul = lambdify(syms[dim], due, 'numpy')
            duq = dul(*X).astype(typecode)
            uf = project(Dx(uh, axis, 1), fft)
            uy = Array(fft)
            uy = fft.backward(uf, uy)
            assert np.allclose(uy, duq, 0, 1e-5)
            for ax in (x for x in range(dim+1) if x is not axis):
                due = ue.diff(xs[axis], 1, xs[ax], 1)
                dul = lambdify(syms[dim], due, 'numpy')
                duq = dul(*X).astype(typecode)
                uf = project(Dx(Dx(uh, axis, 1), ax, 1), fft)
                uy = Array(fft)
                uy = fft.backward(uf, uy)
                assert np.allclose(uy, duq, 0, 1e-5)
                uw = project(dul, fft)
                assert np.allclose(uw, uf, 0, 1e-5)
                uw = project(due, fft)
                assert np.allclose(uw, uf, 0, 1e-5)

            bases.pop(axis)
            fft.destroy()

lagbases_and_quads = list(product(lagBasis, lagquads))
@pytest.mark.parametrize('typecode', 'dD')
@pytest.mark.parametrize('dim', (1, 2))
@pytest.mark.parametrize('ST,quad', lagbases_and_quads)
def test_project_lag(typecode, dim, ST, quad):
    # Using sympy to compute an analytical solution
    x, y, z = symbols("x,y,z")
    sizes = (24, 23)

    funcs = {
        (1, 0): (cos(4*y)*sin(2*x))*exp(-x),
        (1, 1): (cos(4*x)*sin(2*y))*exp(-y),
        (2, 0): (sin(6*z)*cos(4*y)*sin(2*x))*exp(-x),
        (2, 1): (sin(2*z)*cos(4*x)*sin(2*y))*exp(-y),
        (2, 2): (sin(2*x)*cos(4*y)*sin(2*z))*exp(-z)
        }
    syms = {1: (x, y), 2:(x, y, z)}
    xs = {0:x, 1:y, 2:z}

    for shape in product(*([sizes]*dim)):
        bases = []
        for n in shape[:-1]:
            bases.append(Basis(n, 'F', dtype=typecode.upper()))
        bases.append(Basis(shape[-1], 'F', dtype=typecode))

        for axis in range(dim+1):
            ST0 = ST(3*shape[-1], quad=quad)
            bases.insert(axis, ST0)
            fft = TensorProductSpace(comm, bases, dtype=typecode, axes=axes[dim][axis])
            X = fft.local_mesh(True)
            ue = funcs[(dim, axis)]
            u_h = project(ue, fft)
            ul = lambdify(syms[dim], ue, 'numpy')
            uq = ul(*X).astype(typecode)
            uf = u_h.backward()
            assert np.linalg.norm(uq-uf) < 1e-5
            bases.pop(axis)
            fft.destroy()

hbases_and_quads = list(product(hBasis, hquads))
@pytest.mark.parametrize('typecode', 'dD')
@pytest.mark.parametrize('dim', (1, 2))
@pytest.mark.parametrize('ST,quad', hbases_and_quads)
def test_project_hermite(typecode, dim, ST, quad):
    # Using sympy to compute an analytical solution
    x, y, z = symbols("x,y,z")
    sizes = (24, 23)

    funcs = {
        (1, 0): (cos(4*y)*sin(2*x))*exp(-x**2/2),
        (1, 1): (cos(4*x)*sin(2*y))*exp(-y**2/2),
        (2, 0): (sin(6*z)*cos(4*y)*sin(2*x))*exp(-x**2/2),
        (2, 1): (sin(2*z)*cos(4*x)*sin(2*y))*exp(-y**2/2),
        (2, 2): (sin(2*x)*cos(4*y)*sin(2*z))*exp(-z**2/2)
        }
    syms = {1: (x, y), 2:(x, y, z)}
    xs = {0:x, 1:y, 2:z}

    for shape in product(*([sizes]*dim)):
        bases = []
        for n in shape[:-1]:
            bases.append(Basis(n, 'F', dtype=typecode.upper()))
        bases.append(Basis(shape[-1], 'F', dtype=typecode))

        for axis in range(dim+1):
            ST0 = ST(3*shape[-1], quad=quad)
            bases.insert(axis, ST0)
            fft = TensorProductSpace(comm, bases, dtype=typecode, axes=axes[dim][axis])
            X = fft.local_mesh(True)
            ue = funcs[(dim, axis)]
            u_h = project(ue, fft)
            ul = lambdify(syms[dim], ue, 'numpy')
            uq = ul(*X).astype(typecode)
            uf = u_h.backward()
            assert np.linalg.norm(uq-uf) < 1e-5
            bases.pop(axis)
            fft.destroy()



nbases_and_quads = list(product(lBasis[2:3], lquads))+list(product(cBasis[2:3], cquads))
@pytest.mark.parametrize('typecode', 'dD')
@pytest.mark.parametrize('dim', (1, 2))
@pytest.mark.parametrize('ST,quad', nbases_and_quads)
def test_project2(typecode, dim, ST, quad):
    # Using sympy to compute an analytical solution
    x, y, z = symbols("x,y,z")
    sizes = (22, 21)

    funcx = ((2*np.pi**2*(x**2 - 1) - 1)* cos(2*np.pi*x) - 2*np.pi*x*sin(2*np.pi*x))/(4*np.pi**3)
    funcy = ((2*np.pi**2*(y**2 - 1) - 1)* cos(2*np.pi*y) - 2*np.pi*y*sin(2*np.pi*y))/(4*np.pi**3)
    funcz = ((2*np.pi**2*(z**2 - 1) - 1)* cos(2*np.pi*z) - 2*np.pi*z*sin(2*np.pi*z))/(4*np.pi**3)

    funcs = {
        (1, 0): cos(4*y)*funcx,
        (1, 1): cos(4*x)*funcy,
        (2, 0): sin(6*z)*cos(4*y)*funcx,
        (2, 1): sin(2*z)*cos(4*x)*funcy,
        (2, 2): sin(2*x)*cos(4*y)*funcz
        }
    syms = {1: (x, y), 2:(x, y, z)}
    xs = {0:x, 1:y, 2:z}

    for shape in product(*([sizes]*dim)):
        bases = []
        for n in shape[:-1]:
            bases.append(Basis(n, 'F', dtype=typecode.upper()))
        bases.append(Basis(shape[-1], 'F', dtype=typecode))

        for axis in range(dim+1):
            ST0 = ST(shape[-1], quad=quad)
            bases.insert(axis, ST0)
            # Spectral space must be aligned in nonperiodic direction, hence axes
            fft = TensorProductSpace(comm, bases, dtype=typecode, axes=axes[dim][axis])
            X = fft.local_mesh(True)
            ue = funcs[(dim, axis)]
            ul = lambdify(syms[dim], ue, 'numpy')
            uq = ul(*X).astype(typecode)
            uh = Function(fft)
            uh = fft.forward(uq, uh)
            due = ue.diff(xs[axis], 1)
            dul = lambdify(syms[dim], due, 'numpy')
            duq = dul(*X).astype(typecode)
            uf = project(Dx(uh, axis, 1), fft)
            uy = Array(fft)
            uy = fft.backward(uf, uy)
            assert np.allclose(uy, duq, 0, 1e-5)

            # Test also several derivatives
            for ax in (x for x in range(dim+1) if x is not axis):
                due = ue.diff(xs[ax], 1, xs[axis], 1)
                dul = lambdify(syms[dim], due, 'numpy')
                duq = dul(*X).astype(typecode)
                uf = project(Dx(Dx(uh, ax, 1), axis, 1), fft)
                uy = Array(fft)
                uy = fft.backward(uf, uy)
                assert np.allclose(uy, duq, 0, 1e-5)
            bases.pop(axis)
            fft.destroy()

@pytest.mark.parametrize('quad', lquads)
def test_project_2dirichlet(quad):
    x, y = symbols("x,y")
    ue = (cos(4*y)*sin(2*x))*(1-x**2)*(1-y**2)
    sizes = (18, 17)

    D0 = lbases.ShenDirichletBasis(sizes[0], quad=quad)
    D1 = lbases.ShenDirichletBasis(sizes[1], quad=quad)
    B0 = lbases.Basis(sizes[0], quad=quad)
    B1 = lbases.Basis(sizes[1], quad=quad)

    DD = TensorProductSpace(comm, (D0, D1))
    BD = TensorProductSpace(comm, (B0, D1))
    DB = TensorProductSpace(comm, (D0, B1))
    BB = TensorProductSpace(comm, (B0, B1))

    X = DD.local_mesh(True)
    ul = lambdify((x, y), ue, 'numpy')
    uq = Array(DD)
    uq[:] = ul(*X)
    uh = Function(DD)
    uh = DD.forward(uq, uh)

    dudx_hat = project(Dx(uh, 0, 1), BD)
    dudx = Array(BD)
    dudx = BD.backward(dudx_hat, dudx)
    duedx = ue.diff(x, 1)
    duxl = lambdify((x, y), duedx, 'numpy')
    dx = duxl(*X)
    assert np.allclose(dx, dudx, 0, 1e-5)

    dudy_hat = project(Dx(uh, 1, 1), DB)
    dudy = Array(DB)
    dudy = DB.backward(dudy_hat, dudy)
    duedy = ue.diff(y, 1)
    duyl = lambdify((x, y), duedy, 'numpy')
    dy = duyl(*X)
    assert np.allclose(dy, dudy, 0, 1e-5), np.linalg.norm(dy-dudy)

    us_hat = Function(BB)
    us_hat = project(uq, BB, output_array=us_hat)
    us = Array(BB)
    us = BB.backward(us_hat, us)
    assert np.allclose(us, uq, 0, 1e-5)

    dudxy_hat = project(Dx(us_hat, 0, 1) + Dx(us_hat, 1, 1), BB)
    dudxy = Array(BB)
    dudxy = BB.backward(dudxy_hat, dudxy)
    duedxy = ue.diff(x, 1) + ue.diff(y, 1)
    duxyl = lambdify((x, y), duedxy, 'numpy')
    dxy = duxyl(*X)
    assert np.allclose(dxy, dudxy, 0, 1e-5), np.linalg.norm(dxy-dudxy)

@pytest.mark.parametrize('typecode', 'dD')
@pytest.mark.parametrize('dim', (1, 2))
@pytest.mark.parametrize('ST,quad', all_bases_and_quads)
def test_eval_tensor(typecode, dim, ST, quad):
    # Using sympy to compute an analytical solution
    # Testing for Dirichlet and regular basis
    x, y, z = symbols("x,y,z")
    sizes = (22, 21)

    funcx = {'': (1-x**2)*sin(np.pi*x),
             'Dirichlet': (1-x**2)*sin(np.pi*x),
             'Neumann': (1-x**2)*sin(np.pi*x),
             'Biharmonic': (1-x**2)*sin(2*np.pi*x)}
    funcy = {'': (1-y**2)*sin(np.pi*y),
             'Dirichlet': (1-y**2)*sin(np.pi*y),
             'Neumann': (1-y**2)*sin(np.pi*y),
             'Biharmonic': (1-y**2)*sin(2*np.pi*y)}
    funcz = {'': (1-z**2)*sin(np.pi*z),
             'Dirichlet': (1-z**2)*sin(np.pi*z),
             'Neumann': (1-z**2)*sin(np.pi*z),
             'Biharmonic': (1-z**2)*sin(2*np.pi*z)}

    funcs = {
        (1, 0): cos(2*y)*funcx[ST.boundary_condition()],
        (1, 1): cos(2*x)*funcy[ST.boundary_condition()],
        (2, 0): sin(6*z)*cos(4*y)*funcx[ST.boundary_condition()],
        (2, 1): sin(2*z)*cos(4*x)*funcy[ST.boundary_condition()],
        (2, 2): sin(2*x)*cos(4*y)*funcz[ST.boundary_condition()]
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
            bases.append(Basis(n, 'F', dtype=typecode.upper()))
        bases.append(Basis(shape[-1], 'F', dtype=typecode))

        for axis in range(dim+1):
            #for axis in (0,):
            ST0 = ST(shape[-1], quad=quad)
            bases.insert(axis, ST0)
            # Spectral space must be aligned in nonperiodic direction, hence axes
            fft = TensorProductSpace(comm, bases, dtype=typecode, axes=axes[dim][axis])
            print('axes', axes[dim][axis])
            print('bases', bases)
            #print(bases[0].axis, bases[1].axis)
            X = fft.local_mesh(True)
            ue = funcs[(dim, axis)]
            ul = lambdify(syms[dim], ue, 'numpy')
            uu = ul(*X).astype(typecode)
            uq = ul(*points).astype(typecode)
            u_hat = Function(fft)
            u_hat = fft.forward(uu, u_hat)
            t0 = time()
            result = fft.eval(points, u_hat, method=0)
            t_0 += time()-t0
            assert np.allclose(uq, result, 0, 1e-6)
            t0 = time()
            result = fft.eval(points, u_hat, method=1)
            t_1 += time()-t0
            assert np.allclose(uq, result, 0, 1e-6)
            t0 = time()
            result = fft.eval(points, u_hat, method=2)
            t_2 += time()-t0
            print(uq)
            assert np.allclose(uq, result, 0, 1e-6), uq/result
            result = u_hat.eval(points)
            assert np.allclose(uq, result, 0, 1e-6)
            ua = u_hat.backward()
            assert np.allclose(uu, ua, 0, 1e-6)
            ua = Array(fft)
            ua = u_hat.backward(ua)
            assert np.allclose(uu, ua, 0, 1e-6)

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
    sizes = (20, 19)

    funcs = {
        2: cos(4*x) + sin(6*y),
        3: sin(6*x) + cos(4*y) + sin(8*z)
        }
    syms = {2: (x, y), 3:(x, y, z)}
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
            bases.append(Basis(n, 'F', dtype=typecode.upper()))

        bases.append(Basis(shape[-1], 'F', dtype=typecode))
        fft = TensorProductSpace(comm, bases, dtype=typecode)
        X = fft.local_mesh(True)
        ue = funcs[dim]
        ul = lambdify(syms[dim], ue, 'numpy')
        uu = ul(*X).astype(typecode)
        uq = ul(*points).astype(typecode)
        u_hat = Function(fft)
        u_hat = fft.forward(uu, u_hat)
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

    print('method=0', t_0)
    print('method=1', t_1)
    print('method=2', t_2)

if __name__ == '__main__':
    #test_transform('f', 3)
    #test_transform('d', 2)
    #test_shentransform('d', 2, cbases.ShenNeumannBasis, 'GC')
    #test_project('d', 1, cbases.ShenDirichletBasis, 'GC')
    #test_project2('d', 1, lbases.ShenNeumannBasis, 'LG')
    #test_project_2dirichlet('GL')
    test_eval_tensor('d', 2, cbases.ShenBiharmonicBasis, 'GC')
    #test_eval_fourier('d', 3)
