#!/usr/bin/env bash
# The p150a column for the seven perf-page rows the page does not have yet.
# Protocol is frozen in ~/.coworker/state/perf-page-all-models.md §7: byte-pinned prot512.seq,
# shipped defaults, --fast OFF, one card under the device lease, cold pass discarded, warm median
# over n>=3, and the executed batch asserted rather than assumed.
#
# Two benchlock holds, not eight. qb2 runs a 3-4 deep benchlock queue most of the day, so eight
# separate acquisitions spend more wall-clock queueing than measuring. The embed rows go in one
# hold (six short runs) and the two folds in another, which keeps each hold in the same range as
# what the other timed tasks on this box hold.
set -u
WT=/home/ttuser/.coworker/wt/perf-page-tt-cells
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3
export TT_BIO_LEASE_HOLDER=worker:perf-page-tt-cells
export PYTHONPATH="$WT"
export PROBE_ARCH=blackhole-p150a
[ -d /home/ttuser/esm ] && export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_WAIT_S=2400
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
OUT=$WT/perf/wh-embed/results/pp512
SEQ=$WT/scripts/gpu_vs_tt/fixtures/prot512.seq
SHA=141f7d4730ccf17e116016edc4aceee502d8c9769301ece4d1b64beb496ebf8d
export PY BL OUT SEQ SHA WT
mkdir -p "$OUT"

# embeds: model n_seqs assert_batch warmup, one line each
embed_all() {
  while read -r m n b w; do
    [ -n "$m" ] || continue
    [ -s "$OUT/$m.json" ] && { echo "SKIP $m (result present)"; continue; }
    echo "=== embed $m n_seqs=$n assert_batch=$b $(date -u +%H:%M:%SZ)"
    "$PY" -u perf/wh-embed/embed_split_screen.py --model "$m" \
      --n-seqs "$n" --batch-size 8 --assert-batch "$b" \
      --seq-file "$SEQ" --expect-sha "$SHA" --warmup "$w" --repeat 3 \
      --out "$OUT/$m.json"
    echo "EXIT_$m=$?"
  done <<ROWS
saprot-35m 7 7 2
esmc-300m 7 7 2
esmc-600m 7 7 2
saprot-650m 7 7 2
esmc-6b 1 1 1
saprot-1.3b 7 7 2
ROWS
}

fold_all() {
  # esmfold2-fast is the new row. esmfold2 is the same-day control: the denominator for "Fast is
  # faster", and the one measured input to the staleness read on the published 29.393 s cell.
  for m in esmfold2-fast esmfold2; do
    [ -s "$OUT/fold_$m.json" ] && { echo "SKIP fold_$m (result present)"; continue; }
    echo "=== fold $m rounds=3 $(date -u +%H:%M:%SZ)"
    "$PY" -u perf/esm3p4land/fold_ab.py --model "$m" --size 512 --rounds 3 --arms ship \
      --out "$OUT/fold_$m.json"
    echo "EXIT_fold_$m=$?"
  done
}

# A benchlock timeout (75) is "retry later", never "measure anyway". Retry the group instead of
# dropping its rows: a skipped row is a missing cell and the whole point of the run.
for attempt in 1 2 3; do
  echo "### embed group, benchlock attempt $attempt $(date -u +%H:%M:%SZ)"
  "$BL" perf-page-tt-cells -- bash -uc "$(declare -f embed_all); embed_all"
  rc=$?; echo "EMBED_GROUP_RC=$rc"
  [ "$rc" -eq 75 ] || break
done
for attempt in 1 2 3; do
  echo "### fold group, benchlock attempt $attempt $(date -u +%H:%M:%SZ)"
  "$BL" perf-page-tt-cells -- bash -uc "$(declare -f fold_all); fold_all"
  rc=$?; echo "FOLD_GROUP_RC=$rc"
  [ "$rc" -eq 75 ] || break
done
echo "ALL_DONE $(date -u +%H:%M:%SZ)"
