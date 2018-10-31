#!/usr/bin/env python

import os
import sys
import re
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from numpy import get_include

cwd = os.path.abspath(os.path.dirname(__file__))
cdir = os.path.join(cwd, "shenfun", "optimization")

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

def get_extensions():
    ext = []
    for s in ("Matvec", "la", "evaluate"):
        ext.append(Extension("shenfun.optimization.{0}".format(s),
                            libraries=['m'],
                            sources=[os.path.join(cdir, '{0}.pyx'.format(s))],
                            language="c++"))  # , define_macros=define_macros
    [e.extra_link_args.extend(["-std=c++11"]) for e in ext]
    #[e.extra_link_args.extend(["-std=c++11", "-fopenmp"]) for e in ext]
    for s in ("Cheb", "convolve"):
        ext.append(Extension("shenfun.optimization.{0}".format(s),
                            libraries=['m'],
                            sources=[os.path.join(cdir, '{0}.pyx'.format(s))]))
    [e.include_dirs.extend([get_include()]) for e in ext]
    return ext

def version():
    srcdir = os.path.join(cwd, 'shenfun')
    with open(os.path.join(srcdir, '__init__.py')) as f:
        m = re.search(r"__version__\s*=\s*'(.*)'", f.read())
        return m.groups()[0]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="shenfun",
      version=version(),
      description="Shenfun -- Automated Spectral-Galerkin framework",
      long_description=long_description,
      author="Mikael Mortensen",
      author_email="mikaem@math.uio.no",
      url='https://github.com/spectralDNS/shenfun',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Education',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],
      packages=["shenfun",
                "shenfun.optimization",
                "shenfun.legendre",
                "shenfun.chebyshev",
                "shenfun.fourier",
                "shenfun.forms",
                "shenfun.utilities",
                "shenfun.io"
               ],
      package_dir={"shenfun": "shenfun"},
      install_requires=["mpi4py-fft", "mpi4py", "numpy", "scipy"],
      setup_requires=["numpy>=1.11",
                      "cython>=0.25",
                      "setuptools>=18.0"],
      ext_modules=get_extensions(),
      cmdclass={"build_ext": build_ext_subclass}
     )
