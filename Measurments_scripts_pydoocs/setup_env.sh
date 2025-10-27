#!/bin/bash

source /Users/petracon/Desktop/kostas/miniforge3/bin/activate
export DOOCSROOT=/Users/petracon/Desktop/kostas/doocs
export DOOCSARCH=Darwin-aarch64
export PKG_CONFIG_PATH=${DOOCSROOT}/${DOOCSARCH}/lib/pkgconfig
export PYTHONPATH=$PYTHONPATH:$DOOCSROOT/pydoocs/builddir
