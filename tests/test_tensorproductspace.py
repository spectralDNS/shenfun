from __future__ import print_function
import pytest
import numpy as np
from mpi4py import MPI
from shenfun.tensorproductspace import TensorProductSpace
from shenfun.fourier.bases import R2CBasis, C2CBasis

abstol = dict(f=2e-3, d=1e-10, g=1e-12)

def allclose(a, b):
    atol = abstol[a.dtype.char.lower()]
    return np.allclose(a, b, rtol=0, atol=atol)

def random_like(array):
    shape = array.shape
    dtype = array.dtype
    return np.random.random(shape).astype(dtype)

sizes = (9, 12)
@pytest.mark.parametrize('typecode', 'fFdD')
@pytest.mark.parametrize('dim', (2, 3, 4))
def test_transform(typecode, dim):
    from itertools import product

    comm = MPI.COMM_WORLD
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

        fft = TensorProductSpace(comm, bases, dtype=typecode)

        if comm.rank == 0:
            grid = [c.size for c in fft.subcomm]
            print('grid:{} shape:{} typecode:{}'
                    .format(grid, shape, typecode))

        U = random_like(fft.forward.input_array)

        if 1:
            F = fft.forward(U)
            V = fft.backward(F)
            assert allclose(V, U)
        else:
            fft.forward.input_array[...] = U
            fft.forward()
            fft.backward()
            V = fft.backward.output_array
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

        if 1:
            Fc = F.copy()
            V = fft.backward(F)
            F = fft.forward(V)
            assert allclose(F, Fc)
        else:
            fft.backward.input_array[...] = F
            fft.backward()
            fft.forward()
            V = fft.forward.output_array
            assert allclose(F, V)

        fft.destroy()


if __name__ == '__main__':
    test_transform()
