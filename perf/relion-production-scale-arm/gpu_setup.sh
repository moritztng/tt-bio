#!/bin/bash
# Bring a bare CUDA image up to two stock RELION builds (CUDA and ALTCPU) at the commit qb1 runs.
set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential cmake git wget libfftw3-dev libtiff-dev libpng-dev \
  openmpi-bin libopenmpi-dev python3 > /root/apt.log 2>&1
cd /root
tar xzf /root/e6_relion_src.tgz
tar xzf /root/e6_data.tgz
tar xzf /root/prod_stars.tgz
ls -d /root/relion /root/Tutorial5.0/Prod

mkdir -p /root/relion/build-gpu && cd /root/relion/build-gpu
cmake .. -DCUDA=ON -DCudaTexture=OFF -DGUI=OFF -DFETCH_WEIGHTS=OFF -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX=/root/relion/build-gpu \
  > /root/cmake_gpu.log 2>&1
make -j 32 > /root/make_gpu.log 2>&1
test -x /root/relion/build-gpu/bin/relion_refine_mpi && echo GPU_BUILD_OK

mkdir -p /root/relion/build-cpu && cd /root/relion/build-cpu
cmake .. -DALTCPU=ON -DCUDA=OFF -DGUI=OFF -DMKLFFT=OFF -DFETCH_WEIGHTS=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX=/root/relion/build-cpu \
  > /root/cmake_cpu.log 2>&1
make -j 32 > /root/make_cpu.log 2>&1
test -x /root/relion/build-cpu/bin/relion_refine_mpi && echo CPU_BUILD_OK
echo SETUP_DONE > /root/setup.done
