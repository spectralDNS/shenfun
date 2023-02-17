import os
import functools
import numpy as np
from mpi4py import MPI
import pytest
#from mpi4py_fft import generate_xdmf
from shenfun import FunctionSpace, TensorProductSpace, ShenfunFile, Function,\
    Array, CompositeSpace, VectorSpace, generate_xdmf

N = (12, 13, 14, 15)
comm = MPI.COMM_WORLD

skip = {'hdf5': False, 'netcdf4': False}
try:
    import h5py
except ImportError:
    skip['hdf5'] = True
try:
    import netCDF4
except ImportError:
    skip['netcdf4'] = True

ex = {True: 'c', False: 'r'}

writer = functools.partial(ShenfunFile, mode='w')
reader = functools.partial(ShenfunFile, mode='r')

def cleanup():
    import glob
    files = glob.glob('*.h5')+glob.glob('*.xdmf')+glob.glob('*.nc')
    for f in files:
        try:
            os.remove(f)
        except:
            pass

@pytest.mark.parametrize('forward_output', (True, False))
@pytest.mark.parametrize('backend', ('hdf5', 'netcdf4'))
def test_regular_2D(backend, forward_output):
    if (backend == 'netcdf4' and forward_output is True) or skip[backend]:
        return
    K0 = FunctionSpace(N[0], 'F')
    K1 = FunctionSpace(N[1], 'C', bc=(0, 0))
    T = TensorProductSpace(comm, (K0, K1))
    filename = 'test2Dr_{}'.format(ex[forward_output])
    hfile = writer(filename, T, backend=backend)
    u = Function(T, val=1) if forward_output else Array(T, val=1)
    hfile.write(0, {'u': [u]}, forward_output=forward_output)
    hfile.write(1, {'u': [u]}, forward_output=forward_output)
    if not forward_output and backend == 'hdf5' and comm.Get_rank() == 0:
        generate_xdmf(filename+'.h5')
        #generate_xdmf(filename+'.h5', order='visit')

    u0 = Function(T) if forward_output else Array(T)
    read = reader(filename, T, backend=backend)
    read.read(u0, 'u', forward_output=forward_output, step=1)
    assert np.allclose(u0, u)
    T.destroy()
    #cleanup()


@pytest.mark.parametrize('forward_output', (True, False))
@pytest.mark.parametrize('backend', ('hdf5', 'netcdf4'))
@pytest.mark.parametrize('as_scalar', (True, False))
def test_mixed_2D(backend, forward_output, as_scalar):
    if (backend == 'netcdf4' and forward_output is True) or skip[backend]:
        return
    K0 = FunctionSpace(N[0], 'F')
    K1 = FunctionSpace(N[1], 'C')
    T = TensorProductSpace(comm, (K0, K1))
    TT = CompositeSpace([T, T])
    filename = 'test2Dm_{}'.format(ex[forward_output])
    hfile = writer(filename, TT, backend=backend)
    if forward_output:
        uf = Function(TT, val=2)
    else:
        uf = Array(TT, val=2)
    hfile.write(0, {'uf': [uf]}, as_scalar=as_scalar)
    hfile.write(1, {'uf': [uf]}, as_scalar=as_scalar)
    if not forward_output and backend == 'hdf5' and comm.Get_rank() == 0:
        generate_xdmf(filename+'.h5')
    if as_scalar is False:
        u0 = Function(TT) if forward_output else Array(TT)
        read = reader(filename, TT, backend=backend)
        read.read(u0, 'uf', step=1)
        assert np.allclose(u0, uf)
    else:
        u0 = Function(T) if forward_output else Array(T)
        read = reader(filename, T, backend=backend)
        read.read(u0, 'uf0', step=1)
        assert np.allclose(u0, uf[0])
    T.destroy()
    cleanup()

@pytest.mark.parametrize('forward_output', (True, False))
@pytest.mark.parametrize('backend', ('hdf5', 'netcdf4'))
def test_regular_3D(backend, forward_output):
    if (backend == 'netcdf4' and forward_output is True) or skip[backend]:
        return
    K0 = FunctionSpace(N[0], 'F', dtype='D', domain=(0, np.pi))
    K1 = FunctionSpace(N[1], 'F', dtype='d', domain=(0, 2*np.pi))
    K2 = FunctionSpace(N[2], 'C', dtype='d', bc=(0, 0))
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
    if not forward_output and backend == 'hdf5' and comm.Get_rank() == 0:
        generate_xdmf(filename+'.h5')

    u0 = Function(T) if forward_output else Array(T)
    read = reader(filename, T, backend=backend)
    read.read(u0, 'u', forward_output=forward_output, step=1)
    assert np.allclose(u0, u)
    T.destroy()
    #cleanup()

@pytest.mark.parametrize('forward_output', (True, False))
@pytest.mark.parametrize('backend', ('hdf5', 'netcdf4'))
@pytest.mark.parametrize('as_scalar', (True, False))
def test_mixed_3D(backend, forward_output, as_scalar):
    if (backend == 'netcdf4' and forward_output is True) or skip[backend]:
        return
    K0 = FunctionSpace(N[0], 'F', dtype='D', domain=(0, np.pi))
    K1 = FunctionSpace(N[1], 'F', dtype='d', domain=(0, 2*np.pi))
    K2 = FunctionSpace(N[2], 'C')
    T = TensorProductSpace(comm, (K0, K1, K2))
    TT = VectorSpace(T)
    filename = 'test3Dm_{}'.format(ex[forward_output])
    hfile = writer(filename, TT, backend=backend)
    uf = Function(TT, val=2) if forward_output else Array(TT, val=2)
    uf[0] = 1
    data = {'ux': (uf[0],
                   (uf[0], [slice(None), 4, slice(None)]),
                   (uf[0], [slice(None), 4, 4])),
            'uy': (uf[1],
                   (uf[1], [slice(None), 4, slice(None)]),
                   (uf[1], [slice(None), 4, 4])),
            'u': [uf, (uf, [slice(None), 4, slice(None)])]}
    hfile.write(0, data, as_scalar=as_scalar)
    hfile.write(1, data, as_scalar=as_scalar)
    if not forward_output and backend == 'hdf5' and comm.Get_rank() == 0:
        generate_xdmf(filename+'.h5')

    if as_scalar is False:
        u0 = Function(TT) if forward_output else Array(TT)
        read = reader(filename, TT, backend=backend)
        read.read(u0, 'u', step=1)
        assert np.allclose(u0, uf)
    else:
        u0 = Function(T) if forward_output else Array(T)
        read = reader(filename, T, backend=backend)
        read.read(u0, 'u0', step=1)
        assert np.allclose(u0, uf[0])
    T.destroy()
    cleanup()


if __name__ == '__main__':
    for bnd in ('hdf5', 'netcdf4'):
        test_regular_2D(bnd, False)
        test_regular_3D(bnd, False)
        #test_mixed_2D(bnd, False, True)
        #test_mixed_3D(bnd, False, True)
