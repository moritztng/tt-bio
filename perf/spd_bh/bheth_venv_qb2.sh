#!/bin/bash
# spd-bheth: private Python 3.10 venv on qb2 (system python is 3.12, the patched wheel is cp310). Same packages as
# ~/tt-bio-dev/env minus ttnn and the editable tt-bio, then ttnn 0.68.0+bh.eth2 --no-deps. Low priority, download-bound.
set -euo pipefail
B=/home/ttuser/spd-bheth; export UV_PYTHON_INSTALL_DIR=$B/py UV_CACHE_DIR=$B/uvcache
[ -x $B/uvbin/bin/uv ] || { python3 -m venv $B/uvbin && $B/uvbin/bin/pip install -q uv; }
UV=$B/uvbin/bin/uv
$UV python install 3.10
[ -x $B/venv/bin/python ] || $UV venv -p 3.10 $B/venv
$UV pip install -p $B/venv/bin/python torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
# tt-bio deps, constrained to the qb1 venv pins (Python 3.10; qb2 env pins need 3.11+), with the patched wheel as ttnn
$UV pip install -p $B/venv/bin/python -c $B/cons.txt -e $B/treeb "ttnn @ file://$B/ttnn-0.68.0+bh.eth2-cp310-cp310-manylinux_2_34_x86_64.whl" --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
cd $B/treeb && PYTHONPATH=$PWD $B/venv/bin/python -c "import ttnn, tt_bio.metal_overlay as m; print(\"supported\", m.bh_eth_dispatch_supported())" 2>&1 | tail -2
echo VENV_OK
