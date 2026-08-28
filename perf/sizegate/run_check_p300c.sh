#!/bin/bash
# Full size-ladder check, no model subset. tt-smi lives in the tenstorrent venv on qb2 and is not
# on the default PATH, so without this export card detection silently falls back to sysfs.
WT=/home/ttuser/.coworker/wt/protenix-v1-sizeladder-baseline
cd $WT
export PATH=/home/ttuser/.tenstorrent-venv/bin:$PATH
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:protenix-v1-sizeladder-baseline
exec /home/ttuser/tt-bio-dev/env/bin/python3 scripts/release_gate.py --model size-ladder \
    > $WT/perf/sizegate/check_p300c.txt 2>&1
