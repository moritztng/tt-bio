#!/bin/bash
# Positive control: does pc card 0 still hang at 505 tokens, the documented
# deterministic repro (9/9 HANG per state/tt-bio-issue9-protenix-hang-p150a.md)?
# If it does, pc is still in the state that produced the 640 aa hangs and the 640
# result stands on its own. If it does NOT, the card's behaviour has changed and
# every "no longer reproduces" reading is confounded.
D=/home/moritz/.coworker/wt/protenix-v2-640aa-hang-char-pre
cd $D || exit 1
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:protenix-v2-640aa-hang-characterize
PY=/home/moritz/tt-bio/env/bin/python3
$PY scripts/protenix_hang_probe.py --rung 505 --trials 5 --card 0 \
    --timeout 300 --gap 25 --out perf/hangprobe/pc_c0_rung505
echo "=== 505 done, now 507 ==="
$PY scripts/protenix_hang_probe.py --rung 507 --trials 3 --card 0 \
    --timeout 300 --gap 25 --out perf/hangprobe/pc_c0_rung507
echo ALLDONE
