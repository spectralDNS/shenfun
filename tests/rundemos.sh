#!/bin/sh
set -e

pushd $PWD/../demo
python poisson1D.py
python poisson2D.py
python poisson3D.py
python poisson2ND.py

python fourier_poisson1D.py
python fourier_poisson2D.py
python fourier_poisson3D.py

python biharmonic1D.py 32 chebyshev
python biharmonic2D.py
python biharmonic3D.py
python biharmonic1D.py 32 legendre
python biharmonic2D.py legendre
python biharmonic3D.py legendre
python biharmonic1D.py 32 jacobi
python biharmonic2D.py jacobi
python biharmonic3D.py jacobi

python biharmonic2D_2nonperiodic.py
python biharmonic3D_2nonperiodic.py
python biharmonic2D_2nonperiodic.py legendre
python biharmonic3D_2nonperiodic.py legendre

python laguerre_poisson1D.py 70 dirichlet
python laguerre_poisson1D.py 70 neumann
python laguerre_poisson2D.py 60 dirichlet
python laguerre_poisson2D.py 60 neumann

python laguerre_legendre_poisson2D.py 60

python biharmonic2D_2nonperiodic.py chebyshev
python biharmonic2D_2nonperiodic.py legendre
python biharmonic2D_2nonperiodic.py jacobi

python hermite_poisson1D.py 36
python hermite_poisson2D.py 36

python order6.py

python beam_biharmonic1D.py 36

python dirichlet_dirichlet_poisson2D.py 24 25 legendre
python dirichlet_dirichlet_poisson2D.py 24 25 chebyshev
python dirichlet_dirichlet_poisson2D.py 24 25 jacobi

python sphere_helmholtz.py
python curvilinear_poisson1D.py

python NavierStokes.py
python NavierStokesDrivenCavity.py

python MixedPoisson.py 24 25 legendre
python MixedPoisson.py 24 25 chebyshev
python MixedPoisson3D.py legendre
python MixedPoisson3D.py chebyshev
python MixedPoisson1D.py legendre
python MixedPoisson1D.py chebyshev
python Stokes.py legendre
python Stokes.py chebyshev

if ! [ "$(uname)" == "Darwin" ]; then
mpirun -np 4 python poisson2D.py
mpirun -np 4 python poisson3D.py

mpirun -np 4 python fourier_poisson2D.py
mpirun -np 4 python fourier_poisson3D.py

mpirun -np 4 python biharmonic2D.py
mpirun -np 4 python biharmonic3D.py
mpirun -np 4 python biharmonic2D.py legendre
mpirun -np 4 python biharmonic3D.py legendre
mpirun -np 4 python NavierStokes.py
mpirun -np 4 python MixedPoisson.py 24 25 legendre
mpirun -np 4 python MixedPoisson.py 24 25 chebyshev
mpirun -np 4 python MixedPoisson3D.py legendre
mpirun -np 4 python MixedPoisson3D.py chebyshev
mpirun -np 4 python laguerre_poisson2D.py 70 dirichlet
mpirun -np 4 python hermite_poisson2D.py 36
fi

pushd $PWD/../tests
