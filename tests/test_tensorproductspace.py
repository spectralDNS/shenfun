from __future__ import print_function
import pytest
import numpy as np
from mpi4py import MPI
from shenfun.tensorproductspace import TensorProductSpace, VectorTensorProductSpace, \
    MixedTensorProductSpace
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun.chebyshev import bases as cbases
from shenfun.legendre import bases as lbases
from shenfun import inner, div, grad, Function, project, Dx, Array
from sympy import symbols, cos, sin, exp, lambdify
from itertools import product

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
    from itertools import product

    s = (True,)
    if comm.Get_size() > 2 and dim > 2:
        s = (True, False)

    for slab in s:
        for shape in product(*([sizes]*dim)):
            bases = []
            for s in shape[:-1]:
                bases.append(C2CBasis(s))

            if typecode in 'fd':
                bases.append(R2CBasis(shape[-1]))
            else:
                bases.append(C2CBasis(shape[-1]))

            if dim < 3:
                n = min(shape)
                if typecode in 'fdg':
                    n //=2; n+=1
                if n < comm.size:
                    continue

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

            TT = VectorTensorProductSpace([fft,]*dim)
            U = Array(TT, False)
            V = Array(TT, False)
            F = Array(TT)
            U[:] = random_like(U)
            F = TT.forward(U, F)
            V = TT.backward(F, V)
            assert allclose(V, U)

            TM = MixedTensorProductSpace([fft, fft])
            U = Array(TM, False)
            V = Array(TM, False)
            F = Array(TM)
            U[:] = random_like(U)
            F = TM.forward(U, F)
            V = TM.backward(F, V)
            assert allclose(V, U)

            fft.destroy()

            padding = 1.5
            bases = []
            for s in shape[:-1]:
                bases.append(C2CBasis(s, padding_factor=padding))

            if typecode in 'fd':
                bases.append(R2CBasis(shape[-1], padding_factor=padding))
            else:
                bases.append(C2CBasis(shape[-1], padding_factor=padding))

            if dim < 3:
                n = min(shape)
                if typecode in 'fdg':
                    n //=2; n+=1
                if n < comm.size:
                    continue

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

cquads = ('GC', 'GL')
lquads = ('LG', 'GL')

all_bases_and_quads = list(product(lBasis, lquads))+list(product(cBasis, cquads))

@pytest.mark.parametrize('typecode', 'dD')
@pytest.mark.parametrize('dim', (1, 2))
@pytest.mark.parametrize('ST,quad', all_bases_and_quads)
def test_shentransform(typecode, dim, ST, quad):

    for shape in product(*([sizes]*dim)):
        bases = []
        for s in shape[:-1]:
            bases.append(C2CBasis(s))

        if typecode in 'fd':
            bases.append(R2CBasis(shape[-1]))
        else:
            bases.append(C2CBasis(shape[-1]))

        if dim < 3:
            n = min(shape)
            if typecode in 'fdg':
                n //=2; n+=1
            if n < comm.size:
                continue
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
            1: [1,0]}}

@pytest.mark.parametrize('typecode', 'dD')
@pytest.mark.parametrize('dim', (1, 2))
@pytest.mark.parametrize('ST,quad', bases_and_quads)
def test_project(typecode, dim, ST, quad):
    # Using sympy to compute an analytical solution
    x, y, z = symbols("x,y,z")
    sizes=(25, 24)

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
        for s in shape[:-1]:
            bases.append(C2CBasis(s))

        if typecode in 'fd':
            bases.append(R2CBasis(shape[-1]))
        else:
            bases.append(C2CBasis(shape[-1]))

        if dim < 3:
            n = min(shape)
            if typecode in 'fdg':
                n //=2; n+=1
            if n < comm.size:
                continue
        for axis in range(dim+1):
            ST0 = ST(shape[-1], quad=quad)
            bases.insert(axis, ST0)
            fft = TensorProductSpace(comm, bases, dtype=typecode, axes=axes[dim][axis])
            X = fft.local_mesh(True)
            ue = funcs[(dim, axis)]
            ul = lambdify(syms[dim], ue, 'numpy')
            uq = ul(*X).astype(typecode)
            uh = fft.as_function(uq)
            due = ue.diff(xs[axis], 1)
            dul = lambdify(syms[dim], due, 'numpy')
            duq = dul(*X).astype(typecode)
            uf = project(Dx(uh, axis, 1), fft)
            uy = Function(fft, False)
            uy = fft.backward(uf, uy)
            assert np.allclose(uy, duq, 0, 1e-6)
            for ax in (x for x in range(dim+1) if x is not axis):
                due = ue.diff(xs[axis], 1, xs[ax], 1)
                dul = lambdify(syms[dim], due, 'numpy')
                duq = dul(*X).astype(typecode)
                uf = project(Dx(Dx(uh, axis, 1), ax, 1), fft)
                uy = Function(fft, False)
                uy = fft.backward(uf, uy)
                assert np.allclose(uy, duq, 0, 1e-6)

            bases.pop(axis)
            fft.destroy()

#test_project('d', 1, lBasis[0], 'LG')

nbases_and_quads = list(product(lBasis[2:3], lquads))+list(product(cBasis[2:3], cquads))
@pytest.mark.parametrize('typecode', 'dD')
@pytest.mark.parametrize('dim', (1, 2))
@pytest.mark.parametrize('ST,quad', nbases_and_quads)
def test_project2(typecode, dim, ST, quad):
    # Using sympy to compute an analytical solution
    x, y, z = symbols("x,y,z")
    sizes=(25, 24)

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
        for s in shape[:-1]:
            bases.append(C2CBasis(s))

        if typecode in 'fd':
            bases.append(R2CBasis(shape[-1]))
        else:
            bases.append(C2CBasis(shape[-1]))

        if dim < 3:
            n = min(shape)
            if typecode in 'fdg':
                n //=2; n+=1
            if n < comm.size:
                continue
        for axis in range(dim+1):
            ST0 = ST(shape[-1], quad=quad)
            bases.insert(axis, ST0)
            # Spectral space must be aligned in nonperiodic direction, hence axes
            fft = TensorProductSpace(comm, bases, dtype=typecode, axes=axes[dim][axis])
            X = fft.local_mesh(True)
            ue = funcs[(dim, axis)]
            ul = lambdify(syms[dim], ue, 'numpy')
            uq = ul(*X).astype(typecode)
            uh = fft.as_function(uq)
            due = ue.diff(xs[axis], 1)
            dul = lambdify(syms[dim], due, 'numpy')
            duq = dul(*X).astype(typecode)
            uf = project(Dx(uh, axis, 1), fft)
            uy = Function(fft, False)
            uy = fft.backward(uf, uy)
            assert np.allclose(uy, duq, 0, 1e-6)

            # Test also several derivatives
            for ax in (x for x in range(dim+1) if x is not axis):
                due = ue.diff(xs[ax], 1, xs[axis], 1)
                dul = lambdify(syms[dim], due, 'numpy')
                duq = dul(*X).astype(typecode)
                uf = project(Dx(Dx(uh, ax, 1), axis, 1), fft)
                uy = Function(fft, False)
                uy = fft.backward(uf, uy)
                assert np.allclose(uy, duq, 0, 1e-6)
            bases.pop(axis)
            fft.destroy()

@pytest.mark.parametrize('quad', lquads)
def test_project_2dirichlet(quad):
    x, y, z = symbols("x,y,z")
    ue = (cos(4*y)*sin(2*x))*(1-x**2)*(1-y**2)
    sizes = (25, 24)

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
    uq = Function(DD, False)
    uq[:] = ul(*X)

    dudx_hat = project(Dx(uq, 0, 1), BD)
    dudx = Function(BD)
    dudx = BD.backward(dudx_hat, dudx)
    duedx = ue.diff(x, 1)
    duxl = lambdify((x, y), duedx, 'numpy')
    dx = duxl(*X)
    assert np.allclose(dx, dudx)

    dudy_hat = project(Dx(uq, 1, 1), DB)
    dudy = Function(DB, False)
    dudy = DB.backward(dudy_hat, dudy)
    duedy = ue.diff(y, 1)
    duyl = lambdify((x, y), duedy, 'numpy')
    dy = duyl(*X)
    assert np.allclose(dy, dudy)

    us_hat = project(uq, BB)
    us = Function(BB, False)
    us = BB.backward(us_hat, us)
    assert np.allclose(us, uq)

    dudxy_hat = project(Dx(us, 0, 1) + Dx(us, 1, 1), BB)
    dudxy = Function(BB)
    dudxy = BB.backward(dudxy_hat, dudxy)
    duedxy = ue.diff(x, 1) + ue.diff(y, 1)
    duxyl = lambdify((x, y), duedxy, 'numpy')
    dxy = duxyl(*X)
    assert np.allclose(dxy, dudxy)


if __name__ == '__main__':
    test_transform('f', 4)
    #test_transform('d', 2)
    #test_shentransform('d', 2, cbases.ShenNeumannBasis, 'GC')
    #test_project('d', 2, cbases.ShenDirichletBasis, 'GL')
    #test_project2('D', 2, lbases.ShenNeumannBasis, 'GL')
    #test_project_2dirichlet('GL')
