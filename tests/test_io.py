import numpy as np
from mpi4py import MPI
import pytest
import functools
from mpi4py_fft import generate_xdmf
from shenfun import *
from shenfun import ShenfunFile

N = (12, 13, 14, 15)
comm = MPI.COMM_WORLD

skip = False
try:
    import h5py
except ImportError:
    skip = True

ex = {True: 'c', False: 'r'}

writer = functools.partial(ShenfunFile, mode='w')
reader = functools.partial(ShenfunFile, mode='r')

@pytest.mark.skipif(skip, reason='h5py not installed')
@pytest.mark.parametrize('forward_output', (True, False))
@pytest.mark.parametrize('backend', ('hdf5', 'netcdf'))
def test_regular_2D(backend, forward_output):
    if backend == 'netcdf' and forward_output is True:
        return
    K0 = Basis(N[0], 'F')
    K1 = Basis(N[1], 'C')
    T = TensorProductSpace(comm, (K0, K1))
    filename = 'test2Dr_{}'.format(ex[forward_output])
    hfile = writer(filename, T, backend=backend)
    u = Function(T, val=1) if forward_output else Array(T, val=1)
    hfile.write(0, {'u': [u]}, forward_output=forward_output)
    hfile.write(1, {'u': [u]}, forward_output=forward_output)
    hfile.close()
    if not forward_output and backend == 'hdf5' and comm.Get_rank() == 0:
        generate_xdmf(filename+'.h5')

    u0 = Function(T) if forward_output else Array(T)
    read = reader(filename, T, backend=backend)
    read.read(u0, 'u', forward_output=forward_output, step=1)
    assert np.allclose(u0, u)

@pytest.mark.skipif(skip, reason='h5py not installed')
@pytest.mark.parametrize('forward_output', (True, False))
@pytest.mark.parametrize('backend', ('hdf5', 'netcdf'))
@pytest.mark.parametrize('as_scalar', (True, False))
def test_mixed_2D(backend, forward_output, as_scalar):
    if backend == 'netcdf' and forward_output is True:
        return
    K0 = Basis(N[0], 'F')
    K1 = Basis(N[1], 'C')
    T = TensorProductSpace(comm, (K0, K1))
    TT = MixedTensorProductSpace([T, T])
    filename = 'test2Dm_{}'.format(ex[forward_output])
    hfile = writer(filename, TT, backend=backend)
    if forward_output:
        uf = Function(TT, val=2)
    else:
        uf = Array(TT, val=2)
    hfile.write(0, {'uf': [uf]}, forward_output=forward_output, as_scalar=as_scalar)
    hfile.write(1, {'uf': [uf]}, forward_output=forward_output, as_scalar=as_scalar)
    hfile.close()
    if not forward_output and backend == 'hdf5' and comm.Get_rank() == 0:
        generate_xdmf(filename+'.h5')

    if as_scalar is False:
        u0 = Function(TT) if forward_output else Array(TT)
        read = reader(filename, TT, backend=backend)
        read.read(u0, 'uf', forward_output=forward_output, step=1)
        assert np.allclose(u0, uf)
    else:
        u0 = Function(T) if forward_output else Array(T)
        read = reader(filename, T, backend=backend)
        read.read(u0, 'uf0', forward_output=forward_output, step=1)
        assert np.allclose(u0, uf[0])

@pytest.mark.skipif(skip, reason='h5py not installed')
@pytest.mark.parametrize('forward_output', (True, False))
@pytest.mark.parametrize('backend', ('hdf5', 'netcdf'))
def test_regular_3D(backend, forward_output):
    if backend == 'netcdf' and forward_output is True:
        return
    K0 = Basis(N[0], 'F', dtype='D', domain=(0, np.pi))
    K1 = Basis(N[1], 'F', dtype='D', domain=(0, 2*np.pi))
    K2 = Basis(N[2], 'F', dtype='d')
    T = TensorProductSpace(comm, (K0, K1, K2))
    filename = 'test3Dr_{}'.format(ex[forward_output])
    hfile = writer(filename, T, backend=backend)
    if forward_output:
        u = Function(T)
    else:
        u = Array(T)
    u[:] = np.random.random(u.shape)
    for i in range(3):
        hfile.write(i, {'u': [u,
                              (u, [slice(None), 4, slice(None)]),
                              (u, [slice(None), 4, 4])]}, forward_output=forward_output)
    hfile.close()
    if not forward_output and backend == 'hdf5' and comm.Get_rank() == 0:
        generate_xdmf(filename+'.h5')

    u0 = Function(T) if forward_output else Array(T)
    read = reader(filename, T, backend=backend)
    read.read(u0, 'u', forward_output=forward_output, step=1)
    assert np.allclose(u0, u)

@pytest.mark.skipif(skip, reason='h5py not installed')
@pytest.mark.parametrize('forward_output', (True, False))
@pytest.mark.parametrize('backend', ('hdf5', 'netcdf'))
@pytest.mark.parametrize('as_scalar', (True, False))
def test_mixed_3D(backend, forward_output, as_scalar):
    if backend == 'netcdf' and forward_output is True:
        return
    K0 = Basis(N[0], 'F', dtype='D', domain=(0, np.pi))
    K1 = Basis(N[1], 'F', dtype='d', domain=(0, 2*np.pi))
    K2 = Basis(N[2], 'C')
    T = TensorProductSpace(comm, (K0, K1, K2))
    TT = VectorTensorProductSpace(T)
    filename = 'test3Dm_{}'.format(ex[forward_output])
    hfile = writer(filename, TT, backend=backend)
    uf = Function(TT, val=2) if forward_output else Array(TT, val=2)
    uf[0] = 1
    data = {'ux': ((uf, np.s_[0, :, :, :]),
                   (uf, [0, slice(None), 4, slice(None)]),
                   (uf, [0, slice(None), 4, 4])),
            'uy': ((uf, np.s_[1, :, :, :]),
                   (uf, [1, slice(None), 4, slice(None)]),
                   (uf, [1, slice(None), 4, 4])),
            'u': [uf, (uf, [slice(None), slice(None), 4, slice(None)])]}
    hfile.write(0, data, forward_output=forward_output, as_scalar=as_scalar)
    hfile.write(1, data, forward_output=forward_output, as_scalar=as_scalar)
    hfile.close()
    if not forward_output and backend == 'hdf5' and comm.Get_rank() == 0:
        generate_xdmf(filename+'.h5')

    if as_scalar is False:
        u0 = Function(TT) if forward_output else Array(TT)
        read = reader(filename, TT, backend=backend)
        read.read(u0, 'u', forward_output=forward_output, step=1)
        assert np.allclose(u0, uf)
    else:
        u0 = Function(T) if forward_output else Array(T)
        read = reader(filename, T, backend=backend)
        read.read(u0, 'u0', forward_output=forward_output, step=1)
        assert np.allclose(u0, uf[0])

if __name__ == '__main__':
    test_regular_2D('hdf5', False)
    #test_regular_3D('netcdf', False)
    #test_mixed_2D('hdf5', False, False)
    #test_mixed_3D('netcdf', False, False)
