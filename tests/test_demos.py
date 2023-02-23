import subprocess
import os
import pytest
import runpy

demos = [
    'poisson1D',
    'poisson2D',
    'poisson3D',
    'poisson2ND',
    'poisson1D_tau',
    'fourier_poisson1D',
    'fourier_poisson2D',
    'fourier_poisson3D',
    'biharmonic1D',
    'biharmonic2D',
    'biharmonic3D',
    'biharmonic2D_2nonperiodic',
    'biharmonic3D_2nonperiodic',
    'laguerre_poisson1D',
    'laguerre_poisson2D',
    'laguerre_legendre_poisson2D',
    'hermite_poisson1D',
    'hermite_poisson2D',
    'order6',
    'beam_biharmonic1D',
    'dirichlet_dirichlet_poisson2D',
    'sphere_helmholtz',
    'curvilinear_poisson1D',
    'NavierStokes',
    'NavierStokesDrivenCavity',
    'MixedPoisson',
    'MixedPoisson3D',
    'MixedPoisson1D',
    'Stokes'
]

@pytest.mark.skip('skipping MPI')
def test_parallel_demos():
    subprocess.check_output("/bin/bash rundemos.sh", shell=True,
                            cwd=os.path.dirname(os.path.abspath(__file__)))

def test_demos():
    for demo in demos:
        try:
            runpy.run_path(f'demo/{demo}.py', run_name='__main__')
        except SystemExit:
            pass

if __name__ == '__main__':
    test_demos()
