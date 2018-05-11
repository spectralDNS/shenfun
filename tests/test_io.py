import pytest
from shenfun import *
from mpi4py import MPI

N = (12, 13, 14, 15)
comm = MPI.COMM_WORLD

def test_regular_2D():
    K0 = Basis(N[0], 'F')
    K1 = Basis(N[1], 'C')
    T = TensorProductSpace(comm, (K0, K1))
    hfile = HDF5Writer('h5test.h5', ['u'], T)
    u = Function(T, False, val=1)
    hfile.write_tstep(0, u)
    hfile.close()
    generate_xdmf('h5test.h5')

def test_mixed_2D():
    K0 = Basis(N[0], 'F')
    K1 = Basis(N[1], 'C')
    T = TensorProductSpace(comm, (K0, K1))
    TT = MixedTensorProductSpace([T, T])
    hfile = HDF5Writer('h5test2.h5', ['u', 'f'], TT)
    uf = Function(TT, False, val=2)
    hfile.write_tstep(0, uf)
    hfile.close()
    generate_xdmf('h5test2.h5')

def test_regular_3D():
    K0 = Basis(N[0], 'F', dtype='D')
    K1 = Basis(N[1], 'F', dtype='d')
    K2 = Basis(N[2], 'C')
    T = TensorProductSpace(comm, (K0, K1, K2))
    hfile = HDF5Writer('h5test3.h5', ['u'], T)
    u = Function(T, False, val=1)
    hfile.write_tstep(0, u)
    hfile.write_slice_tstep(0, [slice(None), 4, slice(None)], u)
    hfile.write_slice_tstep(0, [slice(None), 4, 4], u)
    hfile.close()
    generate_xdmf('h5test3.h5')

def test_mixed_3D():
    K0 = Basis(N[0], 'F', dtype='D')
    K1 = Basis(N[1], 'F', dtype='d')
    K2 = Basis(N[2], 'C')
    T = TensorProductSpace(comm, (K0, K1, K2))
    TT = MixedTensorProductSpace([T, T])
    hfile = HDF5Writer('h5test4.h5', ['u', 'f'], TT)
    uf = Function(TT, False, val=2)
    hfile.write_tstep(0, uf)
    hfile.write_slice_tstep(0, [slice(None), 4, slice(None)], uf)
    hfile.write_slice_tstep(0, [slice(None), 4, 4], uf)
    hfile.close()
    generate_xdmf('h5test4.h5')

