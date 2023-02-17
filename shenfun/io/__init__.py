import sys
import os
import numpy as np
from mpi4py import MPI
from mpi4py_fft.io import NCFile, HDF5File
from .generate_xdmf import generate_xdmf

__all__ = ['HDF5File', 'NCFile', 'ShenfunFile', 'Checkpoint', 'generate_xdmf']

comm = MPI.COMM_WORLD

def ShenfunFile(name, T, backend='hdf5', mode='r', mesh='quadrature', **kw):
    """Return a file handler

    Parameters
    ----------
    name : str
        Name of file, without ending
    T : :class:`.TensorProductSpace`
        The space used for the data stored. Can also be
        :class:`.CompositeSpace` or
        :class:`.VectorSpace`.
    backend : str, optional
        ``hdf5`` or ``netcdf4``. Default is ``hdf5``.
    mode : str, optional
        ``r`` or ``w``. Default is ``r``.
    mesh : str, optional
        - 'quadrature' - use quadrature mesh of self
        - 'uniform' - use uniform mesh for non-periodic bases

    Returns
    -------
    Class instance
        Instance of either :class:`.HDF5File` or :class:`.NCFile`
    """
    if backend.lower() == 'hdf5':
        return HDF5File(name+'.h5', domain=[np.squeeze(d) for d in T.mesh(kind=mesh)], mode=mode, **kw)
    assert kw.get('forward_output', False) is False, "NetCDF4 cannot store complex arrays, use HDF5"
    return NCFile(name+'.nc', domain=[np.squeeze(d) for d in T.mesh(kind=mesh)], mode=mode, **kw)

class Checkpoint:
    """Class for checkpointing simulations

    Checkpoint data are used to store intermediate simulation results, and can
    be used to restart a simulation at a later stage, with no loss of accuracy.

    Data is provided as dictionaries. The checkpoint dictionary is represented
    as::

        data = {
                '0': {'U': [U_hat], 'phi': [phi_hat]},
                '1': {'U': [U0_hat], 'phi': [phi_hat0]},
                 ...
                }

    which contains solutions to be stored at possibly several different timesteps.
    The current timestep is 0, previous is 1 and so on if more is needed by the
    integrator. Note that checkpoint is storing results from spectral space.

    """
    def __init__(self, filename, checkevery=10, data={}):
        self.f = None
        self.filename = filename
        self.data = data
        self.checkevery = checkevery

    def open(self, mode='r+'):
        import h5py
        self.f = h5py.File(self.filename+'.chk.h5', mode, driver="mpio", comm=comm)

    def close(self):
        if self.f:
            self.f.close()

    def update(self, t, tstep):
        if self.f is None:
            self.open(mode='w')
            self.f.attrs.create('tstep', 0)
            self.f.attrs.create('t', 0.0)
            self.close()

        kill = self.check_if_kill()
        if tstep % self.checkevery == 0 or kill:
            if comm.Get_rank() == 0: # for safety
                os.system(f'cp {self.filename}.chk.h5 {self.filename}.old.chk.h5')
            self.open()
            for key, val in self.data.items():
                self.write(int(key), val)
            self.f.attrs['tstep'] = tstep
            self.f.attrs['t'] = t
            self.close()
            if kill:
                sys.exit(1)

    def write(self, step, d):
        for name, val in d.items():
            self.f.require_group(name)
            for u in val:
                s = u.local_slice()
                self.f[name].require_dataset(str(step), shape=u.global_shape, dtype=u.dtype)
                self.f["/".join((name, str(step)))][s] = u

    def read(self, u, name, **kw):
        step = kw.get('step', 0)
        self.open()
        s = u.local_slice()
        dset = "/".join((name, str(step)))
        u[:] = self.f[dset][s]
        self.close()

    @staticmethod
    def check_if_kill():
        """Check if user has put a file named killshenfun in running folder."""
        found = 0
        if 'killshenfun' in os.listdir(os.getcwd()):
            found = 1
        collective = comm.allreduce(found)
        if collective > 0:
            if comm.Get_rank() == 0:
                os.remove('killshenfun')
                print('killshenfun Found! Stopping simulations cleanly by checkpointing...')
            return True
        else:
            return False
