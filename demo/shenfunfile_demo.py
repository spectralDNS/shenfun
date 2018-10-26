"""Simple demo program for IO with shenfun"""

from mpi4py import MPI
from shenfun import *
from mpi4py_fft import generate_xdmf

N = (24, 25, 26)
backend = 'hdf5'
nsteps = 1
K0 = Basis(N[0], 'F', dtype='D')
K1 = Basis(N[1], 'F', dtype='D')
K2 = Basis(N[2], 'F', dtype='d')
T = TensorProductSpace(MPI.COMM_WORLD, (K0, K1, K2))
TT = MixedTensorProductSpace([T, T])
TV = VectorTensorProductSpace(T)
file_s = ShenfunFile('myfile', T, backend=backend, mode='w')
file_m = ShenfunFile('mixfile', TT, backend=backend, mode='w')
file_v = ShenfunFile('vecfile', TV, backend=backend, mode='w')

u = Array(T)
uf = Array(TT)
U = Array(TV)
tstep = 0
while tstep < nsteps:
    file_s.write(tstep, {'u': [u,
                               (u, [0, slice(None), slice(None)]),
                               (u, [slice(None), 0, slice(None)]),
                               (u, [5, 5, slice(None)])]})

    file_m.write(tstep, {'uf': [uf,
                                (uf, [1, 4, slice(None), slice(None)]),
                                (uf, [slice(None), 0, slice(None), slice(None)]),
                                (uf, [slice(None), 5, 5, slice(None)])]}, as_scalar=True)

    file_v.write(tstep, {'U': [U,
                               (U, [0, 4, slice(None), slice(None)]),
                               (U, [2, 4, slice(None), slice(None)]),
                               (U, [slice(None), 0, slice(None), slice(None)]),
                               (U, [slice(None), 5, 5, slice(None)])],
                         'u': [u]}, as_scalar=False) # A scalar in the vector component space T
    tstep += 1

file_s.close()
file_m.close()
file_v.close()

if backend == 'hdf5':
    generate_xdmf('myfile.h5')
    generate_xdmf('mixfile.h5')
    generate_xdmf('vecfile.h5')

N = (8, 24, 25, 26)
K0 = Basis(N[0], 'F', dtype='D')
K1 = Basis(N[1], 'F', dtype='D')
K2 = Basis(N[2], 'F', dtype='D')
K3 = Basis(N[3], 'F', dtype='d')

T = TensorProductSpace(MPI.COMM_WORLD, (K0, K1, K2, K3))
TT = MixedTensorProductSpace([T, T])
d4file_s = ShenfunFile('my4Dfile', T, backend=backend, mode='w')
d4file_m = ShenfunFile('mix4Dfile', TT, backend=backend, mode='w')

u = Array(T)
uf = Array(TT)

tstep = 0
while tstep < nsteps:
    d4file_s.write(tstep, {'u': [u,
                                 (u, [0, slice(None), slice(None), slice(None)]),
                                 (u, [slice(None), 0, slice(None), slice(None)]),
                                 (u, [slice(None), slice(None), 5, 5])]})

    d4file_m.write(tstep, {'uf': [uf,
                                  (uf, [0, 0, slice(None), slice(None), slice(None)]),
                                  (uf, [slice(None), 0, 0, slice(None), slice(None)]),
                                  (uf, [slice(None), slice(None), 5, 5, 5])],
                           'u': [u]}, as_scalar=True)
    tstep += 1

d4file_s.close()
d4file_m.close()

if backend == 'hdf5':
    generate_xdmf('my4Dfile.h5')
    generate_xdmf('mix4Dfile.h5')

N = (14, 16)
K0 = Basis(N[0], 'F', dtype='D')
K1 = Basis(N[1], 'F', dtype='d')
T = TensorProductSpace(MPI.COMM_WORLD, (K0, K1))
TT = MixedTensorProductSpace([T, T])
d2file_s = ShenfunFile('my2Dfile', T, backend=backend, mode='w')
d2file_m = ShenfunFile('mix2Dfile', TT, backend=backend, mode='w')

u = Array(T)
uf = Array(TT)

tstep = 0
while tstep < nsteps:
    d2file_s.write(tstep, {'u': [u]})
    d2file_m.write(tstep, {'uf': [uf]})
    tstep += 1

d2file_s.close()
d2file_m.close()
if backend == 'hdf5':
    generate_xdmf('my2Dfile.h5')
    generate_xdmf('mix2Dfile.h5')
