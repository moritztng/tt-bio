#!/usr/bin/env bash
# boltz2 32-vs-64 at 76 aa (8HEL), the size openfold3's arm was measured at. Same alternating
# within-pair order and loadavg stamping as b2ab32v64.sh, which is what makes the 20 aa result
# trustworthy; the only change is the fold input.
set -u
WT=/home/ttuser/.coworker/wt/token-axis-bucketing-unify-all-models
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=${CARD:-0}
PAIRS=${PAIRS:-4}
OUT=$WT/perf/tokenbucket/b2ab76
mkdir -p "$OUT"
cd "$WT" || exit 1
: > "$OUT/readings.tsv"

one () {  # one <tag> <multiple>
  local tag=$1 mult=$2 pre post v
  pre=$(cut -d' ' -f1 /proc/loadavg)
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
      TT_BIO_LEASE_HOLDER=worker:token-axis-bucketing-unify-all-models \
      PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm PATH=/home/ttuser/.local/bin:$PATH \
      TT_BIO_TOKEN_BUCKET_MULTIPLE=$mult \
      "$PY" -u perf/tokenbucket/b2size_ab.py --out "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
  post=$(cut -d' ' -f1 /proc/loadavg)
  v=$(grep -aoE "^\[boltz2\] [0-9.]+ structures/s" "$OUT/$tag.log" | tail -1 | awk '{print $2}')
  printf '%s\t%s\t%s\t%s\t%s\n' "$tag" "$mult" "${v:-NA}" "$pre" "$post" | tee -a "$OUT/readings.tsv"
}

for i in $(seq 0 $((PAIRS-1))); do
  if [ $((i % 2)) -eq 0 ]; then
    one "m32_p$i" 32; one "m64_p$i" 64
  else
    one "m64_p$i" 64; one "m32_p$i" 32
  fi
done
echo "=== done $(date -Is)"
cat "$OUT/readings.tsv"
