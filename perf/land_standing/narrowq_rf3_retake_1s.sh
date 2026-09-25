#!/usr/bin/env bash
# The ONE thing narrow-q still owes: rf3 at 896 aa, four reps, clock sampled at a 1 s cadence,
# on a box that stays under the benchlock ceiling for the WHOLE cell.
#
# The cell itself already reads +9.50 s (1.0994x), A/A floor 1.570 %, effect 6.3x its floor, arms
# not overlapping (`perf/land_standing/out/narrowq_rf3_896_qb2c1.json`, read by
# narrowq_rf3_read.py). It is refused by the INSTRUMENT, not the card: clock_during.py needs the
# longest >=1200 MHz run to CONTAIN the timed fold, and a 3 s cadence proves only 104 s inside a
# 104.6 s fold. Every shortfall is under one sampling interval and every leg peaks at 1350 MHz.
# At 1 s the understatement is at most 2 s on a ~100 s fold, which clears the contract.
#
# boltz2 is the WRONG MODEL for this lever and a pass already burned itself proving that twice.
# At 896 aa boltz2 has 4 heads, the wide candidate 448 fits L1, and the fused path serves without
# the fallback: ON and OFF give served/declined [560, 560], the same reject, and the same CIF
# sha256. rf3 is the cell where `fill_preconditions` declines every call. Do not substitute.
#
# PRE-FLIGHT REFUSES rather than waits. The 02:10Z run waited 190 s for a foreign fold and then
# measured anyway, and the reading it produced is discarded.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=0
SIB=1
cd "$WT" || exit 1
mkdir -p "$OUT"

"$PY" perf/c12_orchestrator/pair_guard/host_quiet.py || {
  echo "PRE-FLIGHT REFUSED: host not quiet. This cell needs ~20 min under the ceiling."; exit 3; }
"$PY" perf/c12_orchestrator/pair_guard/pair_idle.py --card $CARD || {
  echo "PRE-FLIGHT REFUSED: board-pair sibling busy."; exit 3; }
echo "PRE-FLIGHT OK $(date -u +%FT%TZ) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"

export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:land-standing
JL=$OUT/narrowq_rf3_retake1s_contention.jsonl
: > "$JL"
"$PY" perf/pvx_gate_land/sample_contention.py --out "$JL" --interval 1.0 &
SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT

exec 9>/home/ttuser/.coworker/state/benchlock.flock
flock -n 9 || { echo "benchlock held by another row; refusing"; exit 3; }
echo "benchlock acquired $(date -u +%FT%TZ)"

# fold_ab_flip now resets the card after a wedge-kill (scripts/card_recovery.py), so a wedged leg
# costs a leg and a board-pair reset instead of the host.
"$PY" perf/xmsoftmax/fold_ab_flip.py --models rf3 --rungs 896 --reps 4 \
  --flag TT_BIO_TRIATT_NARROW_Q_FALLBACK --off-value 1 --fold-timeout-s 400 \
  --workdir "$OUT/narrowq_rf3_retake1s_work" \
  --out "$OUT/narrowq_rf3_retake1s_896.json"
rc=$?
kill $SAMP 2>/dev/null
echo "EXIT rc=$rc $(date -u +%FT%TZ)"
echo "now score the clock contract:"
echo "  $PY perf/pvx_gate_land/clock_during.py --interval 1.0 $OUT/narrowq_rf3_retake1s_896.json"
