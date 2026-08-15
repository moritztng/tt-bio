#!/bin/sh
# Compile line lifted verbatim from RELION's own
# build-e2e/src/apps/CMakeFiles/refine.dir/link.txt, so the screen links the same
# librelion_lib.a object that the profiled refinement ran.
set -e
R=/home/ttuser/relion-scratch/relion
B=$R/build-e2e
/usr/bin/mpicxx -fPIC -std=c++17 -DTIMING -fopenmp -O3 -DNDEBUG \
  -I$R -I$B -I$R/external/healpix_2.15a \
  nzp_screen.cpp -o nzp_screen \
  $B/lib/librelion_lib.a \
  /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so \
  /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so -ldl \
  /usr/lib/x86_64-linux-gnu/libtbb.so \
  /usr/lib/x86_64-linux-gnu/libtiff.so \
  /usr/lib/x86_64-linux-gnu/libfftw3f.so \
  /usr/lib/x86_64-linux-gnu/libfftw3.so \
  /usr/lib/x86_64-linux-gnu/libpython3.10.so \
  /usr/lib/x86_64-linux-gnu/libpng.so \
  /usr/lib/x86_64-linux-gnu/libjpeg.so
