from shenfun import *
from mpi4py import MPI
N = (24, 25, 26)
K0 = Basis(N[0], 'F', dtype='D')
K1 = Basis(N[1], 'F', dtype='D')
K2 = Basis(N[2], 'F', dtype='d')
T = TensorProductSpace(MPI.COMM_WORLD, (K0, K1, K2))
TT = MixedTensorProductSpace([T, T])
h5file = HDF5Writer('myh5file.h5', ['u'], T)
h5file_m = HDF5Writer('mixh5file.h5', ['u', 'f'], TT)

u = Array(T)
uf = Array(TT)
tstep = 0
while tstep < 10:
    h5file.write_tstep(tstep, u)
    h5file.write_slice_tstep(tstep, [0, slice(None), slice(None)], u)
    h5file.write_slice_tstep(tstep, [slice(None), 21, slice(None)], u)
    h5file.write_slice_tstep(tstep, [0, 5, slice(None)], u)
    h5file.write_slice_tstep(tstep, [0, slice(None), 8], u)
    h5file.write_slice_tstep(tstep, [slice(None), 2, 16], u)

    h5file_m.write_tstep(tstep, uf)
    h5file_m.write_slice_tstep(tstep, [0, slice(None), slice(None)], uf)
    h5file_m.write_slice_tstep(tstep, [slice(None), 21, slice(None)], uf)
    h5file_m.write_slice_tstep(tstep, [0, 5, slice(None)], uf)
    h5file_m.write_slice_tstep(tstep, [0, slice(None), 8], uf)
    h5file_m.write_slice_tstep(tstep, [slice(None), 2, 16], uf)

    tstep += 1

h5file.close()
h5file_m.close()
generate_xdmf('myh5file.h5')
generate_xdmf('mixh5file.h5')

N = (5, 24, 25, 26)
K0 = Basis(N[0], 'F', dtype='D')
K1 = Basis(N[1], 'F', dtype='D')
K2 = Basis(N[2], 'F', dtype='D')
K3 = Basis(N[3], 'F', dtype='d')

T = TensorProductSpace(MPI.COMM_WORLD, (K0, K1, K2, K3))
TT = MixedTensorProductSpace([T, T])
h54dfile = HDF5Writer('my4Dfile.h5', ['u'], T)
h54dfile_m = HDF5Writer('mix4Dh5file.h5', ['u', 'f'], TT)

u = Array(T)
uf = Array(TT)

tstep = 0
while tstep < 10:
    h54dfile.write_tstep(tstep, u)
    h54dfile.write_slice_tstep(tstep, [0, slice(None), slice(None), slice(None)], u)
    h54dfile.write_slice_tstep(tstep, [slice(None), 21, slice(None), 22], u)
    h54dfile.write_slice_tstep(tstep, [0, 5, slice(None), 3], u)
    h54dfile.write_slice_tstep(tstep, [0, slice(None), 8, 4], u)
    h54dfile.write_slice_tstep(tstep, [slice(None), 2, 16, 5], u)

    h54dfile_m.write_tstep(tstep, uf)
    h54dfile_m.write_slice_tstep(tstep, [0, slice(None), slice(None), slice(None)], uf)
    h54dfile_m.write_slice_tstep(tstep, [slice(None), 21, slice(None), 22], uf)
    h54dfile_m.write_slice_tstep(tstep, [0, 5, slice(None), 20], uf)
    h54dfile_m.write_slice_tstep(tstep, [0, slice(None), 8, 4], uf)
    h54dfile_m.write_slice_tstep(tstep, [slice(None), 2, 16, 5], uf)

    tstep += 1

N = (14, 16)
K0 = Basis(N[0], 'F', dtype='D')
K1 = Basis(N[1], 'F', dtype='d')
T = TensorProductSpace(MPI.COMM_WORLD, (K0, K1))
TT = MixedTensorProductSpace([T, T])
h5file = HDF5Writer('my2Dh5file.h5', ['u'], T)
h5file_m = HDF5Writer('mix2Dh5file.h5', ['u', 'f'], TT)

u = Array(T)
uf = Array(TT)

tstep = 0
while tstep < 10:
    h5file.write_tstep(tstep, u)
    h5file.write_slice_tstep(tstep, [slice(None), 2], u)

    h5file_m.write_tstep(tstep, uf)
    h5file_m.write_slice_tstep(tstep, [0, slice(None)], uf)

    tstep += 1

#ncfile = NCWriter('mynetcdf.nc', ['u'], T)
#ncfile.write_tstep(0, u)
#ncfile.write_tstep(1, u)

