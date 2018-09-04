# pylint: disable=line-too-long
import copy
import six
from numpy import dtype
try:
    import h5py
except ImportError:
    import warnings
    warnings.warn('h5py not installed')


__all__ = ('generate_xdmf',)

xdmffile = """<?xml version="1.0" encoding="utf-8"?>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.1">
  <Domain>
    <Grid Name="Structured Grid" GridType="Collection" CollectionType="Temporal">
"""

timeattr = """      <Time TimeType="List"><DataItem Format="XML" Dimensions="{1}"> {0} </DataItem></Time>"""

attribute3D = """
        <Attribute Name="{0}" Center="Node">
          <DataItem Format="HDF" NumberType="Float" Precision="{6}" Dimensions="{1} {2} {3}">
            {4}:/{5}
          </DataItem>
        </Attribute>"""

attribute2D = """
        <Attribute Name="{0}" Center="Node">
          <DataItem Format="HDF" NumberType="Float" Precision="{5}" Dimensions="{1} {2}">
            {3}:/{4}
          </DataItem>
        </Attribute>"""

mesh_3d = """
        <Geometry Type="VXVYVZ">
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{1}">
           {4}:/mesh/x0
          </DataItem>
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{2}">
           {4}:/mesh/x1
          </DataItem>
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{3}">
           {4}:/mesh/x2
          </DataItem>
        </Geometry>"""

mesh_2d = """
        <Geometry Type="VXVY">
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{1}">
           {3}:/mesh/{4}
          </DataItem>
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{2}">
           {3}:/mesh/{5}
          </DataItem>
        </Geometry>"""

def generate_xdmf(h5filename):
    """Generate XDMF-files

    Parameters
    ----------
        h5filename : str
            Name of hdf5-file that we want to decorate with xdmf
    """
    f = h5py.File(h5filename)
    keys = []
    f.visit(keys.append)
    assert 'mesh' in keys

    # Find unique groups
    datasets = {2:{}, 3:{}}  # 2D and 3D datasets
    for key in keys:
        if isinstance(f[key], h5py.Dataset):
            if not 'mesh' in key:
                tstep = int(key.split("/")[-1])
                ndim = len(f[key].shape)
                if ndim in (2, 3):
                    ds = datasets[ndim]
                    if tstep in ds:
                        ds[tstep] += [key]
                    else:
                        ds[tstep] = [key]

    coor = {0:'x0', 1:'x1', 2:'x2'}
    for ndim, dsets in six.iteritems(datasets):
        timesteps = list(dsets.keys())
        if not timesteps:
            continue

        timesteps.sort(key=int)
        xf = copy.copy(xdmffile)
        tt = ""
        for i in timesteps:
            tt += "%s " %i

        xf += timeattr.format(tt, len(timesteps))

        datatype = f[dsets[timesteps[0]][0]].dtype
        prec = 4 if datatype is dtype('float32') else 8
        if ndim == 2:
            xff = {}
            coors = {}
            NN = {}
            for name in dsets[timesteps[0]]:
                slices = name.split("/")[2]
                if not slices in xff:
                    xff[slices] = copy.copy(xf)
                    NN[slices] = f[name].shape
                    if 'slice' in slices:
                        ss = slices.split("_")
                        coors[slices] =  [coor[i] for i, sx in enumerate(ss) if 'slice' in sx]
                    else:
                        coors[slices] = ['x0', 'x1']

            # if slice of 3D data, need to know xy, xz or yz plane
            # Since there may be many different slices, we need to create
            # one xdmf-file for each composition of slices
            for tstep in timesteps:
                d = dsets[tstep]
                for slices in xff.keys():
                    cc = coors[slices]
                    N = NN[slices]
                    xff[slices] += """
      <Grid GridType="Uniform">"""
                    xff[slices] += mesh_2d.format(prec, N[0], N[1], h5filename, cc[0], cc[1])
                    xff[slices] += """
        <Topology Dimensions="{0} {1}" Type="2DRectMesh"/>""".format(*N)

                for i, x in enumerate(d):
                    slices = x.split("/")[2]
                    if not 'slice' in slices:
                        slices = dsets[timesteps[0]][0].split("/")[2]
                    N = NN[slices]
                    xff[slices] += attribute2D.format(x.split("/")[0], N[0], N[1], h5filename, x, prec)

                for slices in xff.keys():
                    xff[slices] += """
      </Grid>"""

            for slices, ff in six.iteritems(xff):
                ff += """
    </Grid>
  </Domain>
</Xdmf>
"""
                fname = h5filename[:-3]+"_"+slices+".xdmf" if 'slice' in slices else h5filename[:-3]+".xdmf"
                xfl = open(fname, "w")
                xfl.write(ff)
                xfl.close()

        elif ndim == 3:
            for tstep in timesteps:
                d = dsets[tstep]
                N = f[d[0]].shape
                xf += """
      <Grid GridType="Uniform">"""
                xf += mesh_3d.format(prec, N[0], N[1], N[2], h5filename)
                xf += """
        <Topology Dimensions="{0} {1} {2}" Type="3DRectMesh"/>""".format(*N)
                for x in d:
                    name = x.split("/")[0]
                    xf += attribute3D.format(name, N[0], N[1], N[2], h5filename, x, prec)
                xf += """
      </Grid>"""
            xf += """
    </Grid>
  </Domain>
</Xdmf>
"""
            xfl = open(h5filename[:-3]+".xdmf", "w")
            xfl.write(xf)
            xfl.close()

if __name__ == "__main__":
    import sys
    generate_xdmf(sys.argv[-1])
