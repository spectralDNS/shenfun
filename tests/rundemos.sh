#!/bin/sh
set -e

pushd $PWD/../demo

if ! [ "$(uname)" == "Darwin" ]; then
mpirun -np 4 python poisson2D.py
mpirun -np 4 python poisson3D.py
mpirun -np 4 python fourier_poisson2D.py
mpirun -np 4 python fourier_poisson3D.py
mpirun -np 4 python biharmonic2D.py
mpirun -np 4 python biharmonic3D.py
mpirun -np 4 python NavierStokes.py
mpirun -np 4 python MixedPoisson.py
mpirun -np 4 python MixedPoisson3D.py
mpirun -np 4 python laguerre_poisson2D.py
mpirun -np 4 python hermite_poisson2D.py
fi

pushd $PWD/../tests
