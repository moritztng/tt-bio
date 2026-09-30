#!/bin/bash
# The design proof: tt-bio from the BUILT WHEEL in a clean 3.12 venv, BindCraft 2 from its own
# checkout the way upstream's README installs it. `pip install` of BindCraft 2 does not carry its
# `settings/` or `examples/` trees (its pyproject packages `bindcraft*` only), so the installed
# copy cannot read `settings/core/default.json`; the checkout on PYTHONPATH is what a user has.
# tt_bio still resolves from site-packages, and the driver prints where from.
cd ~/b2pship
export TT_VISIBLE_DEVICES=2
export BCX_BC2=$HOME/bcx_e2e/bc2
export AF2_PARAMS=$HOME/bcx_e2e/af2_params
export DESIGN_OUT=$HOME/b2pship_design
export BINDER=60
export JAX_COMPILATION_CACHE_DIR=$HOME/b2pship/out/xlacache_wheel
export PYTHONPATH=$HOME/bcx_e2e/bc2
rm -rf "$DESIGN_OUT"
timeout 5400 ~/b2pship_venv312/bin/python -u ~/b2pship/out/design_wheel.py
echo "DESIGN_RC=$? $(date -u +%FT%TZ)"
