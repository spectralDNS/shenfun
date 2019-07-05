#!/bin/bash

if [[ "$(uname)" == "Darwin" ]]; then
  export CXXFLAGS="-std=c++11 -stdlib=libc++ $CXXFLAGS"
  export LDFLAGS="-Wl,-rpath,$PREFIX/lib $LDFLAGS"
  export MACOSX_DEPLOYMENT_TARGET=10.9
fi

pip install -r "${RECIPE_DIR}/component-requirements.txt"
$PYTHON -m pip install . --no-deps --ignore-installed --no-cache-dir -vvv
