#!/bin/bash
set -x
cd ~/b2pship
P=$HOME/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12
rm -rf ~/b2pship_venv312
$P -m venv ~/b2pship_venv312
~/b2pship_venv312/bin/pip install -q --upgrade pip
W=$(ls ~/b2pship_dist/*.whl | head -1)
sha256sum "$W"
~/b2pship_venv312/bin/pip install -q "$W[tenstorrent]" 2>&1 | tail -3
~/b2pship_venv312/bin/pip install -q ~/bcx_e2e/bc2 2>&1 | tail -3
~/b2pship_venv312/bin/python ~/b2pship/out/probe312.py
echo "V312_RC=$? $(date -u +%FT%TZ)"
