#!/usr/bin/env bash
# narrow-q A/B, re-taken on a genuinely quiet box.
#
# The 2026-09-22 02:10Z run is discarded, not repeated: every one of its eight legs ran with a
# foreign `tt_bio.main predict` above 50 % CPU, SFPI kernel compiles landed inside the slowest
# leg, a `tt-smi -r 2` board-pair reset fired mid-run, loadavg swung 1.26-9.44 and
# r(loadavg, runtime) was +0.476. Its benchlock waited 190 s for the foreign fold and then
# acquired anyway. That run's 9.707 % A/A floor and its 768 aa "negative control failed"
# reading are both artifacts of that contention.
#
# PRE-FLIGHT IS A REFUSAL, NOT A WAIT: if a foreign fold exists or the box is loud, this exits
# rather than measuring, because the previous run proves waiting-then-proceeding is the failure.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out
mkdir -p "$OUT"
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=0
SIB=1          # card 0's board-pair sibling on this p300c; the pair shares a power budget
cd "$WT" || exit 1

foreign=$(ps -eo pcpu,args --sort=-pcpu | grep -E "tt_bio\.main|fold_ab|boltzgen|pxdesign" \
          | grep -v grep | grep -v narrowq | awk '$1>20' | head -3)
if [ -n "$foreign" ]; then echo "PRE-FLIGHT REFUSED -- foreign fold running:"; echo "$foreign"; exit 3; fi
la=$(cut -d' ' -f1 /proc/loadavg)
if [ "$(echo "$la > 2.0" | bc)" = "1" ]; then echo "PRE-FLIGHT REFUSED -- loadavg $la > 2.0"; exit 3; fi
for c in $CARD $SIB; do
  n=$(sudo -n lsof /dev/tenstorrent/$c 2>/dev/null | tail -n +2 | wc -l)
  [ "$n" != "0" ] && { echo "PRE-FLIGHT REFUSED -- card $c has $n device fds"; exit 3; }
done
echo "PRE-FLIGHT OK  loadavg $la  cards $CARD/$SIB idle  no foreign fold  $(date -u +%FT%TZ)"

export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:land-standing
JL=$OUT/narrowq_quiet_contention.jsonl
: > "$JL"
$PY perf/pvx_gate_land/sample_contention.py --out "$JL" --interval 1.0 &
SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT

exec 9>/home/ttuser/.coworker/state/benchlock.flock
flock -n 9 || { echo "benchlock held by another row; refusing"; exit 3; }
echo "benchlock acquired $(date -u +%FT%TZ)"

run_cell () {   # <rung> <reps> <out>
  echo "=== rf3 $1 aa, $2 reps  $(date -u +%FT%TZ) ==="
  $PY perf/xmsoftmax/fold_ab_flip.py --models rf3 --rungs "$1" --reps "$2" \
      --flag TT_BIO_TRIATT_NARROW_Q_FALLBACK --off-value 1 --fold-timeout-s 300 \
      --workdir "$OUT/narrowq_quiet_work" --out "$3"
}

run_cell 896 4 "$OUT/narrowq_quiet_896_qb2c0.json"
run_cell 768 2 "$OUT/narrowq_quiet_768_qb2c0.json"
kill $SAMP 2>/dev/null
echo "=== ALL CELLS DONE $(date -u +%FT%TZ) ==="
