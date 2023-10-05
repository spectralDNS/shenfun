#!/bin/bash

if [[ "$(uname)" == "Darwin" ]]; then
  export CXXFLAGS="-std=c++11 -stdlib=libc++ $CXXFLAGS"
  export LDFLAGS="-Wl,-rpath,$PREFIX/lib $LDFLAGS"
  export MACOSX_DEPLOYMENT_TARGET=10.9
fi

export CFLAGS="-DCYTHON_FAST_THREAD_STATE=0 -DCYTHON_USE_EXC_INFO_STACK=0 $CFLAGS"

$PYTHON -m pip install . --no-deps --ignore-installed --no-cache-dir -vvv
