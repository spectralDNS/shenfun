import os
import yaml

# The configuration can be overloaded by a local 'shenfun.yaml' file, or
# in '~/.shenfun/shenfun.yaml'. A yaml file to work with can be created
# using the `dumpconfig` function below

config = {
    'optimization':
    {
        'mode': 'cython',
        'verbose': False,
    },
    'basisvectors': 'normal',
    'matrix':
    {
        'sparse':
        {
            'solve': 'csc',
            'diags': 'csc',
            'matvec': 'csr',
            'construct': 'dense' # denser, sympy - The method used to construct non-implemented matrices
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
    }
}

locations = [os.path.expanduser('~/.shenfun'),
             os.getcwd()]

for loc in locations:
    fl = os.path.expandvars(os.path.join(loc, 'shenfun.yaml'))
    try:
        with open(fl, 'r') as yf:
            config.update(yaml.load(yf, Loader=yaml.FullLoader))
        yf.close()
    except FileNotFoundError:
        pass

def dumpconfig(filename='shenfun.yaml', path='~/.shenfun'): # pragma: no cover
    """Dump a configuration file in yaml format
    """
    with open(os.path.join(os.path.expanduser(path), filename), 'w') as yf:
        yaml.dump(config, yf)
    yf.close()
