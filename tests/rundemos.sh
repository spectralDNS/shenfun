#/bin/sh
set -e

pushd $PWD/../demo
python dirichlet_poisson1D.py 32 chebyshev
python dirichlet_poisson2D.py 32 chebyshev
python dirichlet_poisson3D.py 32 chebyshev
python dirichlet_poisson1D.py 32 legendre
python dirichlet_poisson2D.py 32 legendre
python dirichlet_poisson3D.py 32 legendre

python fourier_poisson1D.py
python fourier_poisson2D.py
python fourier_poisson3D.py

python neumann_poisson1D.py
python neumann_poisson2D.py
python neumann_poisson3D.py
python neumann_poisson1D.py legendre
python neumann_poisson2D.py legendre
python neumann_poisson3D.py legendre

python biharmonic1D.py
python biharmonic2D.py
python biharmonic3D.py
python biharmonic1D.py legendre
python biharmonic2D.py legendre
python biharmonic3D.py legendre

python dirichlet_dirichlet_poisson2D.py

mpirun -np 4 python dirichlet_poisson2D.py 24 chebyshev
mpirun -np 4 python dirichlet_poisson3D.py 24 chebyshev
mpirun -np 4 python dirichlet_poisson2D.py 24 legendre
mpirun -np 4 python dirichlet_poisson3D.py 24 legendre

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

pushd $PWD/../tests
