#!bin/sh
set -e

pushd $PWD/../demo
python dirichlet_poisson1D.py 64 chebyshev
python dirichlet_poisson2D.py 32 chebyshev
python dirichlet_poisson3D.py 32 chebyshev
python dirichlet_poisson1D.py 64 legendre
python dirichlet_poisson2D.py 32 legendre
python dirichlet_poisson3D.py 32 legendre

python dirichlet_Helmholtz1D.py 32 chebyshev
python dirichlet_Helmholtz2D.py 32 chebyshev
python dirichlet_Helmholtz1D.py 32 legendre
python dirichlet_Helmholtz2D.py 32 legendre

python fourier_poisson1D.py
python fourier_poisson2D.py
python fourier_poisson3D.py

python neumann_poisson1D.py
python neumann_poisson2D.py
python neumann_poisson3D.py
python neumann_poisson1D.py legendre
python neumann_poisson2D.py legendre
python neumann_poisson3D.py legendre

python biharmonic1D.py 32 chebyshev
python biharmonic2D.py
python biharmonic3D.py
python biharmonic1D.py 32 legendre
python biharmonic2D.py legendre
python biharmonic3D.py legendre

python biharmonic2D_2nonperiodic.py
python biharmonic3D_2nonperiodic.py
python biharmonic2D_2nonperiodic.py legendre
python biharmonic3D_2nonperiodic.py legendre

python dirichlet_dirichlet_poisson2D.py 24 25 legendre
python dirichlet_dirichlet_poisson2D.py 24 25 chebyshev

python NavierStokes.py

python MixedPoisson.py 24 25 legendre
python MixedPoisson.py 24 25 chebyshev
python Stokes.py legendre
python Stokes.py chebyshev

mpirun -np 4 python dirichlet_poisson2D.py 24 chebyshev
mpirun -np 4 python dirichlet_poisson3D.py 24 chebyshev
mpirun -np 4 python dirichlet_poisson2D.py 24 legendre
mpirun -np 4 python dirichlet_poisson3D.py 24 legendre

mpirun -np 4 python dirichlet_Helmholtz2D.py 32 legendre
mpirun -np 4 python dirichlet_Helmholtz2D.py 32 chebyshev

mpirun -np 4 python fourier_poisson2D.py
mpirun -np 4 python fourier_poisson3D.py

mpirun -np 4 python neumann_poisson2D.py
mpirun -np 4 python neumann_poisson3D.py
mpirun -np 4 python neumann_poisson2D.py legendre
mpirun -np 4 python neumann_poisson3D.py legendre

mpirun -np 4 python biharmonic2D.py
mpirun -np 4 python biharmonic3D.py
mpirun -np 4 python biharmonic2D.py legendre
mpirun -np 4 python biharmonic3D.py legendre
mpirun -np 4 python NavierStokes.py
mpirun -np 4 python MixedPoisson.py 24 25 legendre
mpirun -np 4 python MixedPoisson.py 24 25 chebyshev
pushd $PWD/../tests
