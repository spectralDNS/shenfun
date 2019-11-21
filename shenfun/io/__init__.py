import numpy as np
from mpi4py_fft.io import NCFile, HDF5File

__all__ = ['HDF5File', 'NCFile', 'ShenfunFile']


def ShenfunFile(name, T, backend='hdf5', mode='r', uniform=False, **kw):
    """Return a file handler

    Parameters
    ----------
    name : str
        Name of file, without ending
    T : :class:`.TensorProductSpace`
        The space used for the data stored. Can also be
        :class:`.MixedTensorProductSpace` or
        :class:`.VectorTensorProductSpace`.
    backend : str, optional
        ``hdf5`` or ``netcdf4``. Default is ``hdf5``.
    mode : str, optional
        ``r`` or ``w``. Default is ``r``.
    uniform : bool, optional
        Use uniform mesh for non-periodic bases if True

    Returns
    -------
    Class instance
        Instance of either :class:`.HDF5File` or :class:`.NCFile`
    """
    if backend.lower() == 'hdf5':
        return HDF5File(name+'.h5', domain=[np.squeeze(d) for d in T.mesh(uniform=uniform)], mode=mode, **kw)
    assert kw.get('forward_output', False) is False, "NetCDF4 cannot store complex arrays, use HDF5"
    return NCFile(name+'.nc', domain=[np.squeeze(d) for d in T.mesh(uniform=uniform)], mode=mode, **kw)
