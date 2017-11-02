import copy
import six
from numpy import pi, float32, dtype
try:
    import h5py
except:
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

mesh_3d =  """
        <Geometry Type="VXVYVZ">
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{3}">
           {4}:/mesh/z
          </DataItem>
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{2}">
           {4}:/mesh/y
          </DataItem>
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{1}">
           {4}:/mesh/x
          </DataItem>
        </Geometry>"""

mesh_2d =  """
        <Geometry Type="VXVY">
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{1}">
           {3}:/mesh/{4}
          </DataItem>
          <DataItem Format="HDF" NumberType="Float" Precision="{0}" Dimensions="{2}">
           {3}:/mesh/{5}
          </DataItem>
        </Geometry>"""

def generate_xdmf(h5filename):
    f = h5py.File(h5filename)
    keys = []
    f.visit(keys.append)
    assert 'mesh' in keys
    assert 'mesh/x' in keys
    assert 'mesh/y' in keys

    # Find unique groups
    datasets = {2:{}, 3:{}}  # 2D and 3D datasets
    for key in keys:
        if isinstance(f[key], h5py.Dataset):
            if not 'mesh' in key:
                tstep = eval(key.split("/")[-1])
                ndim = len(f[key].shape)
                ds = datasets[ndim]
                if tstep in ds:
                    ds[tstep] += [key]
                else:
                    ds[tstep] = [key]

    coor = {0:'x', 1:'y', 2:'z'}
    for ndim, dsets in six.iteritems(datasets):
        timesteps = list(dsets.keys())
        if len(timesteps) == 0:
            continue

        timesteps.sort(key=int)
        xf = copy.copy(xdmffile)
        tt = ""
        for i in timesteps:
            tt += "%s " %i

        xf += timeattr.format(tt, len(timesteps))

        datatype = f[dsets[timesteps[0]][0]].dtype
        prec = 4 if datatype is dtype('float32') else 8

        N = f[dsets[timesteps[0]][0]].shape

        if ndim == 2:
            for tstep in timesteps:
                d = dsets[tstep]
                # if slice of 3D data, need to know xy, xz or yz plane
                for x in d:
                    name = x.split("/")[:-1]
                    if 'slice' in name[-1]:
                        ss = name[-1].split('_')
                        coors = [coor[i] for i,sx in enumerate(ss) if 'slice' in sx]
                    else:
                        coors = ['x', 'y']
                    break

                xf += """
      <Grid GridType="Uniform">"""
                xf += mesh_2d.format(prec, N[0], N[1], h5filename, coors[0], coors[1])
                xf += """
        <Topology Dimensions="{0} {1}" Type="2DRectMesh"/>""".format(*N)
                for x in d:
                    name = x.split("/")[:-1]
                    xf += attribute2D.format("_".join(name), N[0], N[1], h5filename, x, prec)
                xf += """
      </Grid>
"""

        elif ndim == 3:
            for tstep in timesteps:
                d = dsets[tstep]
                xf += """
      <Grid GridType="Uniform">"""
                xf += mesh_3d.format(prec, N[0], N[1], N[2], h5filename)
                xf += """
        <Topology Dimensions="{0} {1} {2}" Type="3DRectMesh"/>""".format(*N)
                for x in d:
                    name = x.split("/")[:-1]
                    xf += attribute3D.format("_".join(name), N[0], N[1], N[2], h5filename, x, prec)
                xf += """
      </Grid>
"""
        xf += """
    </Grid>
  </Domain>
</Xdmf>
"""
        xfl = open(h5filename[:-3]+"_"+str(ndim)+"D.xdmf", "w")
        xfl.write(xf)
        xfl.close()

if __name__ == "__main__":
    import sys
    generate_xdmf(sys.argv[-1])
