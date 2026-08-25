#!/usr/bin/env bash
# Does RF3's token bucket cost anything at a REAL size? The 20 aa arm read -10.1% but was taken
# while the rel071 release gate ramped up on card 1 (loadavg 3.5 -> 6.3 across the run), so it is
# not trustworthy on its own. 20 aa is also the size where the bucket can only cost: 20 pads to 32,
# which is the SAME single tile physically, so pad + mask-upload + slice per Pairformer call buy
# no compute back at all.
#
# 76 aa is the test that matters: 76 pads to 96, three tiles, where the alignment actually does
# something. Alternating order makes a busy box common-mode.
set -u
WT=/home/ttuser/.coworker/wt/token-axis-bucketing-unify-all-models
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=${CARD:-0}
PAIRS=${PAIRS:-3}
PRE=b328f757~1
OUT=$WT/perf/tokenbucket/rf3_bucket_ab76
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
      "$PY" -u perf/tokenbucket/b2size_ab.py --model rf3 --input examples/8hel_nomsa.yaml \
      --out "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
  post=$(cut -d' ' -f1 /proc/loadavg)
  v=$(grep -aoE "^\[rf3\] [0-9.]+ structures/s" "$OUT/$tag.log" | tail -1 | awk '{print $2}')
  printf '%s\t%s\t%s\t%s\t%s\n' "$tag" "$arm" "${v:-NA}" "$pre" "$post" | tee -a "$OUT/readings.tsv"
}

for i in $(seq 0 $((PAIRS-1))); do
  if [ $((i % 2)) -eq 0 ]; then one "buck_p$i" buck; one "raw_p$i" raw
  else                          one "raw_p$i" raw;  one "buck_p$i" buck; fi
done
restore_head
echo "=== done $(date -Is)"; cat "$OUT/readings.tsv"
