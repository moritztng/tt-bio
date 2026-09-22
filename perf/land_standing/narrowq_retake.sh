#!/usr/bin/env bash
# The narrow-q re-take pass 17 pre-registered: a 1 s clock cadence, four reps at the size the
# lever fires at, and a 768 aa negative control that must read 1.00x because the policy function
# cannot reach the fallback list there.
#
# It retries a cell once, because this cell wedges. On 2026-09-22 an rf3 896 aa leg stopped at
# "trunk 1/10" with its device child spinning at 100 % CPU and no progress for 11 minutes, on a
# card that had just folded the same shape in 101.7 s. --fold-timeout-s bounds the wedge and
# `tt-smi -r` between attempts clears the card the killed fold left dirty.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out
mkdir -p "$OUT"
JL=$OUT/narrowq_retake_contention.jsonl
: > "$JL"
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=0

export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:land-standing

cd "$WT" || exit 1
$PY perf/pvx_gate_land/sample_contention.py --out "$JL" --interval 1.0 &
SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT

run_cell () {   # <rung> <reps> <out>
  local rung="$1" reps="$2" out="$3" attempt
  for attempt in 1 2; do
    echo "=== rf3 $rung aa, $reps reps, attempt $attempt ==="
    if $PY perf/xmsoftmax/fold_ab_flip.py --models rf3 --rungs "$rung" --reps "$reps" \
        --flag TT_BIO_TRIATT_NARROW_Q_FALLBACK --off-value 1 \
        --fold-timeout-s 300 \
        --workdir "$OUT/narrowq_retake_work" --out "$out"; then
      echo "=== rf3 $rung aa OK on attempt $attempt ==="
      return 0
    fi
    echo "=== rf3 $rung aa FAILED on attempt $attempt; resetting card $CARD ==="
    ~/.local/bin/tt-smi -r "$CARD" >/dev/null 2>&1
  done
  return 1
}

run_cell 896 4 "$OUT/narrowq_retake_896_qb2c0.json"
run_cell 768 2 "$OUT/narrowq_retake_768_qb2c0.json"

kill $SAMP 2>/dev/null
echo "=== ALL ARMS DONE ==="
