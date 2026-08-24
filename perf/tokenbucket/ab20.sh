#!/usr/bin/env bash
# Live paired A/B of the token bucket at the perf_regression gate's own input (examples/trpcage.yaml,
# 20 aa, padded 20 -> 64 -- the most extreme relative pad anywhere). The gate's red reads the flip
# against a COMMITTED p300c baseline, and 27 commits of this branch plus 358 of main sit between the
# two, so the committed delta cannot attribute anything. This measures ON against OFF in the same
# process order on the same card at the same load, which can.
set -u
WT=/home/ttuser/.coworker/wt/tokenbucket-rebase-and-land
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=${CARD:?}
OUT=$WT/perf/tokenbucket/ab20
mkdir -p "$OUT"
cd "$WT" || exit 1

one () {  # one <model> <arm 1|0> <pair>
  local m=$1 arm=$2 pair=$3 tag="$1.bucket$2.p$3"
  [ -s "$OUT/$tag.log" ] && { echo "SKIP $tag"; return 0; }
  echo "$(date -Is) BEGIN $tag load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/progress.log"
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
      TT_BIO_LEASE_HOLDER=worker:tokenbucket-rebase-and-land \
      PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm \
      TT_BIO_PROTENIX_TOKEN_BUCKET=$arm PATH=/home/ttuser/tt-bio/env/bin:$PATH \
      "$PY" -u scripts/perf_regression.py --model "$m" > "$OUT/$tag.log" 2>&1
  echo "$(date -Is) END   $tag rc=$? load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/progress.log"
}

for pair in 1 2; do
  for m in protenix-v2 opendde opendde-abag; do
    one $m 1 $pair
    one $m 0 $pair
  done
done
echo "$(date -Is) ALL DONE" >> "$OUT/progress.log"
