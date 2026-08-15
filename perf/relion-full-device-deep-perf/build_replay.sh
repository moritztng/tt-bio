#!/bin/sh
# Build coarse_replay with a chosen set of extra defines/flags.
#
# The base line is lifted from build-e2e/src/apps/CMakeFiles/relion_lib.dir/flags.make so the
# harness compiles diff2.h with byte-identical macros to the library the profiled refinement ran --
# in particular -DALTCPU=1 -DPROJECTOR_NO_TEXTURES and NO -march, which is what makes the sincosf in
# TRANSLATE_PIXEL_2D scalar.
#
#   $1  output binary name
#   $2+ extra flags (e.g. -DUSE_SINCOS_TABLE, -march=native)
set -e
R=/home/ttuser/relion-scratch/relion
B=$R/build-e2e
OUT=$1; shift
DEFS="-DACC_CPU=1 -DACC_CUDA=2 -DACC_HIP=3 -DALTCPU=1 -DHAVE_JPEG -DHAVE_PNG -DHAVE_SINCOS \
-DHAVE_TIFF -DPROJECTOR_NO_TEXTURES -DUSE_MPI_COLLECTIVE"
/usr/bin/mpicxx -fPIC -std=c++14 -DTIMING -fopenmp -O3 -DNDEBUG $DEFS "$@" \
  -I$R -I$B -I$R/external/healpix_2.15a \
  coarse_replay.cpp -o "$OUT" \
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
echo "built $OUT with: $*"
