#!/bin/bash
# Force-grid A/B at 505, on a freshly reset card, with a timeout that allows a
# cold JIT compile (the prior workstream measured 275s cold for the forced-grid
# fold, so a 300s cap cannot tell "hang" from "still compiling").
D=/home/moritz/.coworker/wt/protenix-v2-640aa-hang-char-pre
cd $D || exit 1
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:protenix-v2-640aa-hang-characterize
PY=/home/moritz/tt-bio/env/bin/python3
echo "### ARM A: forced 11x10 ###"
$PY scripts/protenix_hang_probe.py --rung 505 --trials 3 --card 0 \
    --timeout 900 --gap 25 --force-grid 11,10 \
    --out perf/hangprobe/ab505_g11
echo "### ARM B: default grid (13x10) ###"
$PY scripts/protenix_hang_probe.py --rung 505 --trials 2 --card 0 \
    --timeout 900 --gap 25 \
    --out perf/hangprobe/ab505_default
echo ALLDONE
