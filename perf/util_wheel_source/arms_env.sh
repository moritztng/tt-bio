# util-wheel-vs-source: the two arms. Source ONE of them, never both.
export WT=/home/ttuser/.coworker/wt/util-wheel-vs-source
export TT_VISIBLE_DEVICES=3
export TT_BIO_LEASE_CARDS=3
export TT_BIO_LEASE_HOLDER=worker:util-wheel-vs-source
export PY=/home/ttuser/tt-bio-dev/env/bin/python3
export PATH=/home/ttuser/tt-bio-dev/env/bin:$PATH
export ART=/home/ttuser/scratch/uws_art
mkdir -p $ART
arm_wheel() {   # pip ttnn 0.68.0 out of the venv site-packages
  unset TT_METAL_HOME
  unset LD_LIBRARY_PATH
  export PYTHONPATH=$WT
}
arm_source() {  # tt-metal built from source at the wheel own tag v0.68.0 (1452925b)
  export TT_METAL_HOME=/home/ttuser/tt-metal-b2z
  unset LD_LIBRARY_PATH
  export PYTHONPATH=$WT:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:$TT_METAL_HOME
}
