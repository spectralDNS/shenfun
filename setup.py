#!/usr/bin/env python

import os, sys
from distutils.core import setup, Extension
import subprocess
from numpy import get_include
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from Cython.Compiler.Options import get_directive_defaults

# Version number
major = 1
minor = 1

cwd = os.path.abspath(os.path.dirname(__file__))
cdir = os.path.join(cwd, "shenfun", "optimization")

ext = None
cmdclass = {}
class build_ext_subclass(build_ext):
    def build_extensions(self):
        #extra_compile_args = ['-w', '-Ofast', '-fopenmp', '-march=native']
        extra_compile_args = ['-w', '-Ofast', '-march=native']
        cmd = "echo | %s -E - %s &>/dev/null" % (
            self.compiler.compiler[0], " ".join(extra_compile_args))
        try:
            subprocess.check_call(cmd, shell=True)
        except:
            extra_compile_args = ['-w', '-O3', '-ffast-math', '-march=native']
            #extra_compile_args = ['-w', '-O3', '-ffast-math', '-fopenmp', '-march=native']
        for e in self.extensions:
            e.extra_compile_args += extra_compile_args
        build_ext.build_extensions(self)

args = ""
if not "sdist" in sys.argv:
    if "build_ext" in sys.argv:
        args = "build_ext --inplace"

    ext = []
    for s in ("Matvec", "la"):
        ext += cythonize(Extension("shenfun.optimization.{0}".format(s),
                                   libraries=['m'],
                                   sources=[os.path.join(cdir, '{0}.pyx'.format(s))],
                                   language="c++"))  # , define_macros=define_macros
    [e.extra_link_args.extend(["-std=c++11"]) for e in ext]
    #[e.extra_link_args.extend(["-std=c++11", "-fopenmp"]) for e in ext]

    for s in ("Cheb", "convolve", "evaluate"):
        ext += cythonize(Extension("shenfun.optimization.{0}".format(s),
                                   libraries=['m'],
                                   sources = [os.path.join(cdir, '{0}.pyx'.format(s))]))

    [e.include_dirs.extend([get_include()]) for e in ext]
    cmdclass = {'build_ext': build_ext_subclass}

else:
    pass

setup(name = "shenfun",
      version = "%d.%d" % (major, minor),
      description = "Shenfun -- Automated Spectral-Galerkin framework",
      long_description = "",
      author = "Mikael Mortensen",
      author_email = "mikaem@math.uio.no",
      url = 'https://github.com/spectralDNS/shenfun',
      classifiers = [
          'Development Status :: 5 - Production/Stable',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Education',
          'Programming Language :: Python :: 2.7',
          'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],
      packages = ["shenfun",
                  "shenfun.optimization",
                  "shenfun.legendre",
                  "shenfun.chebyshev",
                  "shenfun.fourier",
                  "shenfun.forms",
                  "shenfun.utilities"
                  ],
      package_dir = {"shenfun": "shenfun"},
      ext_modules = ext,
      cmdclass = cmdclass
    )
