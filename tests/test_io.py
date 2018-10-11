from shenfun import *
from mpi4py import MPI
import numpy as np
import pytest

N = (12, 13, 14, 15)
comm = MPI.COMM_WORLD

skip = False
try:
    import h5py
except ImportError:
    skip = True

ex = {True: 'c', False: 'r'}

@pytest.mark.skipif(skip, reason='h5py not installed')
@pytest.mark.parametrize('forward_output', (True, False))
def test_regular_2D(forward_output):
    K0 = Basis(N[0], 'F')
    K1 = Basis(N[1], 'C')
    T = TensorProductSpace(comm, (K0, K1))
    filename = 'h5test_{}.h5'.format(ex[forward_output])
    hfile = HDF5Writer(filename, ['u'], T, forward_output=forward_output)
    if forward_output:
        u = Function(T, val=1)
    else:
        u = Array(T, val=1)
    hfile.write_tstep(0, u, forward_output)
    hfile.close()
    if not forward_output:
        generate_xdmf(filename)

@pytest.mark.skipif(skip, reason='h5py not installed')
@pytest.mark.parametrize('forward_output', (True, False))
def test_mixed_2D(forward_output):
    K0 = Basis(N[0], 'F')
    K1 = Basis(N[1], 'C')
    T = TensorProductSpace(comm, (K0, K1))
    TT = MixedTensorProductSpace([T, T])
    filename = 'h5test2_{}.h5'.format(ex[forward_output])
    hfile = HDF5Writer(filename, ['u', 'f'], TT, forward_output=forward_output)
    if forward_output:
        uf = Function(TT, val=2)
    else:
        uf = Array(TT, val=2)
    hfile.write_tstep(0, uf, forward_output)
    hfile.close()
    if not forward_output:
        generate_xdmf(filename)

@pytest.mark.skipif(skip, reason='h5py not installed')
@pytest.mark.parametrize('forward_output', (True, False))
def test_regular_3D(forward_output):
    K0 = Basis(N[0], 'F', dtype='D')
    K1 = Basis(N[1], 'F', dtype='D')
    K2 = Basis(N[2], 'F', dtype='d')
    T = TensorProductSpace(comm, (K0, K1, K2))
    filename = 'h5test3_{}.h5'.format(ex[forward_output])
    hfile = HDF5Writer(filename, ['u'], T, forward_output)
    if forward_output:
        u = Function(T)
    else:
        u = Array(T)
    u[:] = np.random.random(u.shape)
    for i in range(3):
        hfile.write_tstep(i, u, forward_output)
        hfile.write_slice_tstep(i, [slice(None), 4, slice(None)], u, forward_output)
        hfile.write_slice_tstep(i, [slice(None), 4, 4], u, forward_output)
    hfile.close()
    if not forward_output:
        generate_xdmf(filename)

@pytest.mark.skipif(skip, reason='h5py not installed')
@pytest.mark.parametrize('forward_output', (True, False))
def test_mixed_3D(forward_output):
    K0 = Basis(N[0], 'F', dtype='D')
    K1 = Basis(N[1], 'F', dtype='d')
    K2 = Basis(N[2], 'C')
    T = TensorProductSpace(comm, (K0, K1, K2))
    TT = MixedTensorProductSpace([T, T])
    filename = 'h5test4_{}.h5'.format(ex[forward_output])
    hfile = HDF5Writer(filename, ['u', 'f'], TT, forward_output=forward_output)
    if forward_output:
        uf = Function(TT, val=2)
    else:
        uf = Array(TT, val=2)
    hfile.write_tstep(0, uf, forward_output)
    hfile.write_slice_tstep(0, [slice(None), 4, slice(None)], uf, forward_output)
    hfile.write_slice_tstep(0, [slice(None), 4, 4], uf, forward_output)
    hfile.close()
    if not forward_output:
        generate_xdmf(filename)

if __name__ == '__main__':
    test_regular_3D(True)
