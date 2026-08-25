#!/usr/bin/env bash
# Does the RF3 token bucket (b328f757) move RF3's perf? The gate cannot answer it: rf3 has NO
# BASELINE for p300c on this box, and seeding one off this branch would bake the change in as its
# own reference -- the one thing a lever's branch must not do.
#
# So measure the change directly. Arm "buck" is HEAD; arm "raw" restores the two files from the
# commit before the bucket, working-tree only, no commits. Same instrument as the gate
# (perf_regression.py, 2 warmup + 5 timed, median), alternating order, loadavg stamped.
set -u
WT=/home/ttuser/.coworker/wt/token-axis-bucketing-unify-all-models
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=${CARD:-0}
PAIRS=${PAIRS:-2}
PRE=b328f757~1
OUT=$WT/perf/tokenbucket/rf3_bucket_ab
mkdir -p "$OUT"
cd "$WT" || exit 1
: > "$OUT/readings.tsv"

restore_head () { git checkout -q HEAD -- tt_bio/rf3/model.py tt_bio/rf3/confidence_head.py; }
trap restore_head EXIT

one () {  # one <tag> <arm>
  local tag=$1 arm=$2 pre post v
  if [ "$arm" = raw ]; then
    git checkout -q $PRE -- tt_bio/rf3/model.py tt_bio/rf3/confidence_head.py
  else
    restore_head
  fi
  pre=$(cut -d' ' -f1 /proc/loadavg)
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
      TT_BIO_LEASE_HOLDER=worker:token-axis-bucketing-unify-all-models \
      PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm PATH=/home/ttuser/.local/bin:$PATH \
      "$PY" -u scripts/perf_regression.py --model rf3 > "$OUT/$tag.log" 2>&1
  post=$(cut -d' ' -f1 /proc/loadavg)
  v=$(grep -aoE "^\[rf3\] [0-9.]+ structures/s" "$OUT/$tag.log" | tail -1 | awk '{print $2}')
  printf '%s\t%s\t%s\t%s\t%s\n' "$tag" "$arm" "${v:-NA}" "$pre" "$post" | tee -a "$OUT/readings.tsv"
}

for i in $(seq 0 $((PAIRS-1))); do
  if [ $((i % 2)) -eq 0 ]; then
    one "buck_p$i" buck; one "raw_p$i" raw
  else
    one "raw_p$i" raw;  one "buck_p$i" buck
  fi
done
restore_head
echo "=== done $(date -Is); tree restored:"; git status --porcelain
cat "$OUT/readings.tsv"
