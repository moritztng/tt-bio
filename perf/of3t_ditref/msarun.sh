#!/usr/bin/env bash
# D58's msa_module arm, re-taken on today's tree. Card 1 on qb2, clock sampled DURING.
#
# The repaired denominator cannot move this: perf/of3t_ditref/ARCHDIFF.json shows the whole
# 0.4.3 -> 0.5.0 boundary is 24 in and 1 out, all inside diffusion_module, and msa_module is
# name-identical at 227 parameters on both trees. What CAN have moved it is the same thing that
# moved D30 from 19.6x to 11.03x with the forward bit-identical: our own backward.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-ditref
cd "$W"
S=/tmp/of3t/of3t-ditref
PY=/home/ttuser/tt-bio-dev/env/bin/python
DEV=1
source "$W/perf/refpath.sh"
B=$REF_BUNDLE
export PYTHONPATH="$(ref_pythonpath "$REF_PYLIBS" "$W/perf/of3t_tape" "$W")"
ref_assert "$PY"
export OMP_NUM_THREADS=4
export TT_VISIBLE_DEVICES=$DEV TT_BIO_LEASE_CARDS=$DEV TT_BIO_LEASE_HOLDER=worker:of3t-ditref
TAG=$1; shift
CLK=$S/aiclk_$TAG.log; LOG=$S/$TAG.log; : > "$CLK"
( while :; do
    echo "$(date +%s) $(cat /sys/class/tenstorrent/tenstorrent\!$DEV/tt_aiclk 2>/dev/null || echo NA)" >> "$CLK"
    sleep 5
  done ) &
SIDE=$!
trap 'kill $SIDE 2>/dev/null' EXIT
L0=$(cut -d' ' -f1-3 /proc/loadavg); T0=$(date +%s)
echo "=== msa $TAG start $(date -u +%FT%TZ) load[$L0] ===" | tee "$LOG"
"$PY" perf/of3t_auxheads/msa_instrument.py \
  --boundary /home/ttuser/of3t_auxheads/cap043b/boundary_msa_module.pt \
  --reference-grads "$B/grads_f64_043.pt" "$@" \
  --out "perf/of3t_ditref/$TAG.json" >> "$LOG" 2>&1
RC=$?; T1=$(date +%s); L1=$(cut -d' ' -f1-3 /proc/loadavg); kill $SIDE 2>/dev/null
CLKSTAT=$(awk '$2!="NA"{print $2}' "$CLK" | sort -n | awk '{a[n++]=$1; s+=$1} END{if(n==0){print "NA";exit} printf "n=%d min=%s med=%s max=%s mean=%.1f", n, a[0], a[int(n/2)], a[n-1], s/n}')
echo "=== msa $TAG exit $RC seconds $((T1-T0)) load_start[$L0] load_end[$L1] aiclk_during{$CLKSTAT} $(date -u +%FT%TZ) ===" | tee -a "$LOG"
printf '%s\n' "$TAG $RC $((T1-T0)) \"$L0\" \"$L1\" \"$CLKSTAT\"" >> "$S/ARMS.tsv"
exit $RC
