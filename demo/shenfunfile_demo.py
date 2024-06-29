"""Simple demo program for IO with shenfun"""

from mpi4py import MPI
from shenfun import *
from mpi4py_fft import generate_xdmf

N = (24, 25, 26)
backend = 'hdf5'
nsteps = 1
K0 = FunctionSpace(N[0], 'F', dtype='D')
K1 = FunctionSpace(N[1], 'F', dtype='D')
K2 = FunctionSpace(N[2], 'F', dtype='d')
T = TensorProductSpace(MPI.COMM_WORLD, (K0, K1, K2))
TT = CompositeSpace([T, T])
TV = VectorSpace(T)
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
                                (uf, [4, slice(None), slice(None)]),
                                (uf, [0, slice(None), slice(None)]),
                                (uf, [5, 5, slice(None)])]}, as_scalar=True)

    file_v.write(tstep, {'U': [U,
                               (U, [4, slice(None), slice(None)]),
                               (U, [4, slice(None), slice(None)]),
                               (U, [0, slice(None), slice(None)]),
                               (U, [5, 5, slice(None)])],
                         'u': [u]}, as_scalar=False) # A scalar in the vector component space T
    tstep += 1

if backend == 'hdf5' and MPI.COMM_WORLD.Get_rank() == 0:
    generate_xdmf('myfile.h5')
    generate_xdmf('mixfile.h5')
    generate_xdmf('vecfile.h5')

N = (8, 24, 25, 26)
K0 = FunctionSpace(N[0], 'F', dtype='D')
K1 = FunctionSpace(N[1], 'F', dtype='D')
K2 = FunctionSpace(N[2], 'F', dtype='D')
K3 = FunctionSpace(N[3], 'F', dtype='d')

T = TensorProductSpace(MPI.COMM_WORLD, (K0, K1, K2, K3))
TT = CompositeSpace([T, T])
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
                                  (uf, [0, slice(None), slice(None), slice(None)]),
                                  (uf, [0, 0, slice(None), slice(None)]),
                                  (uf, [slice(None), 5, 5, 5])],
                           'u': [u]}, as_scalar=True)
    tstep += 1

if backend == 'hdf5' and MPI.COMM_WORLD.Get_rank() == 0:
    generate_xdmf('my4Dfile.h5')
    generate_xdmf('mix4Dfile.h5')

N = (14, 16)
K4 = FunctionSpace(N[0], 'F', dtype='D')
K5 = FunctionSpace(N[1], 'F', dtype='d')
T1 = TensorProductSpace(MPI.COMM_WORLD, (K4, K5))
TT = CompositeSpace([T1, T1])
d2file_s = ShenfunFile('my2Dfile', T1, backend=backend, mode='w')
d2file_m = ShenfunFile('mix2Dfile', TT, backend=backend, mode='w')

u = Array(T1)
uf = Array(TT)

tstep = 0
while tstep < nsteps:
    d2file_s.write(tstep, {'u': [u]})
    d2file_m.write(tstep, {'uf': [uf]})
    tstep += 1

if backend == 'hdf5' and MPI.COMM_WORLD.Get_rank() == 0:
    generate_xdmf('my2Dfile.h5')
    generate_xdmf('mix2Dfile.h5')
cleanup(vars())