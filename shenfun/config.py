import os
from collections.abc import Mapping
from collections import defaultdict
import yaml

# The configuration can be overloaded by a local 'shenfun.yaml' file, or
# in '~/.shenfun/shenfun.yaml'. A yaml file to work with can be created
# using the `dumpconfig` function below
try:
    import numba
    has_numba = True
except:
    has_numba = False

config = {
    'optimization':
    {
        'mode': 'cython',
        'verbose': False,
    },
    'basisvectors': 'normal',
    'transforms':
    {
        'kind': defaultdict(lambda: 'vandermonde',
            {
                'chebyshev': 'fast',
                'fourier': 'fast',
                'legendre': 'fast' if has_numba else 'vandermonde'
            }) # The other families need to have Orthogonal basis overload _evaluate_scalar_product and _evaluate_expansion_all
    },
    'matrix':
    {
        'sparse':
        {
            #'permc_spec': 'NATURAL',
            'permc_spec': 'COLAMD',
            'solve': 'csc',
            'diags': 'csc',
            'matvec': 'csr'
        },
        'block':
        {
            'assemble': 'csc',
            'use_scipy': True,
            'permc_spec': 'COLAMD',
        }
    },
    'bases':
    {
        'legendre':
        {
            'mode': 'numpy',
            'precision': 30,
        },
        'jacobi':
        {
            'mode': 'numpy',
            #'mode': 'mpmath',
            'precision': 30
        }
    },
    'fftw':
    {
        'dct':
        {
            'threads': 1,
            'planner_effort': 'FFTW_MEASURE',
        },
        'dst':
        {
            'threads': 1,
            'planner_effort': 'FFTW_MEASURE',
        },
        'rfft':
        {
            'threads': 1,
            'planner_effort': 'FFTW_MEASURE'
        },
        'irfft':
        {
            'threads': 1,
            'planner_effort': 'FFTW_MEASURE'
        },
        'fft':
        {
            'threads': 1,
            'planner_effort': 'FFTW_MEASURE'
        },
        'ifft':
        {
            'threads': 1,
            'planner_effort': 'FFTW_MEASURE'
        }
    }
}

def update(conf, newconf):
    """Recursive update"""
    for key, value in newconf.items():
        if isinstance(value, Mapping) and value:
            returned = update(conf.get(key, {}), value)
            conf[key] = returned
        else:
            conf[key] = newconf[key]
    return conf

locations = [os.path.expanduser('~/.shenfun'),
             os.getcwd()]

for loc in locations:
    fl = os.path.expandvars(os.path.join(loc, 'shenfun.yaml'))
    try:
        with open(fl, 'r') as yf:
            update(config, yaml.load(yf, Loader=yaml.FullLoader))
        yf.close()
    except FileNotFoundError:
        pass

def dumpconfig(filename='shenfun.yaml', path='~/.shenfun'): # pragma: no cover
    """Dump a configuration file in yaml format
    """
    with open(os.path.join(os.path.expanduser(path), filename), 'w') as yf:
        yaml.dump(config, yf)
    yf.close()
