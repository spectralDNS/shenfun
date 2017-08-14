#!/bin/bash
echo Making sphinx-rootdir
mkdir sphinx-rootdir
sphinx-quickstart <<EOF
sphinx-rootdir
n
_
Shenfun - automating the spectral Galerkin method
Mikael Mortensen
1.0
1.0
en
.rst
index
y
y
n
n
n
n
n
y
n
y
y
y

EOF
