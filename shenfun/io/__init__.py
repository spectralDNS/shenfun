from .h5py_file import HDF5File
from .nc_file import NCFile

__all__ = ['HDF5File', 'NCFile', 'ShenfunFile']


def ShenfunFile(name, T, backend='hdf5', mode='r', **kw):
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

    Returns
    -------
    Class instance
        Instance of either :class:`.HDF5File` or :class:`.NCFile`
    """
    if backend.lower() == 'hdf5':
        return HDF5File(name+'.h5', T, mode=mode, **kw)
    assert kw.get('forward_output', False) is False, "NetCDF4 cannot store complex arrays, use HDF5"
    return NCFile(name+'.nc', T, mode=mode, **kw)
