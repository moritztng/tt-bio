#!/usr/bin/env bash
# The 768 aa sweep, on pc card 0. The one leg the brief names that no host has completed:
# qb2 got 7 of 9 folds before it hard-hung (state doc 14a).
#
# Both arms are measured HERE. The committed 100.95 s is a qb2 number and a pc arm compared to it
# would be denominator drift (rule 4 of the 2026-08-25 amendment). a1's own warm spread on this
# card is the A/A control and the only thing an arm is allowed to be compared against.
set -u
WT=/home/moritz/.coworker/wt/rf3-fused-hifi-precision-arm
PY=/home/moritz/tt-bio/env/bin/python3
PP=$WT:/home/moritz/rf3_perf_deps
BL=/home/moritz/.coworker/scripts/benchlock.sh
R=$WT/perf/rf3/results
L=$R/logs
cd "$WT"
LEASE="TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:rf3-fused-hifi-precision-arm"

# benchlock gates on loadavg before the run and does not hold it there; both 13b and 14a caught a
# co-tenant inside a timed window. Record uptime either side so a polluted fold is identifiable.
echo "=== $(date -Is) START hifi_ckc_768_pc  uptime: $(uptime)" >> "$L/legs_pc.log"
bash "$BL" rf3-hifi-768-pc -- env PYTHONPATH=$PP $LEASE \
  "$PY" -u perf/rf3/hifi_ckc_ab.py --aa 768 --arms a1,a5,a9 --sweeps 3 \
  --out "$R/hifi_ckc_768_pc.json" > "$L/hifi_ckc_768_pc.log" 2>&1
rc=$?
echo "=== $(date -Is) END hifi_ckc_768_pc rc=$rc  uptime: $(uptime)" >> "$L/legs_pc.log"
