#!/usr/bin/env bash
# Pad-multiple sweep at 20 aa on p300c, both models that pay it. The parent task ran this on p150a
# and read "pad 32 recovers 2.6 of 16 points", i.e. mostly a per-call cost. p300c reads a much
# bigger flip cost (-30.5 % on protenix-v2, -11.6 % on opendde against a live OFF arm in the same
# hour), so the attribution has to be re-taken on this card rather than carried over: 20 -> 32 is a
# quarter of 20 -> 64's padded pair area, so if the cost is area, m32 recovers ~3/4 of it, and if it
# is per-call, m32 recovers almost none.
set -u
WT=/home/ttuser/.coworker/wt/tokenbucket-rebase-and-land
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=${CARD:?}
OUT=$WT/perf/tokenbucket/padsweep_p300c
mkdir -p "$OUT"
cd "$WT" || exit 1
LEASE="TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:tokenbucket-rebase-and-land"

arm () {  # arm <model> <tag> <env...>
  local m=$1 tag=$2; shift 2
  [ -s "$OUT/$m.$tag.log" ] && { echo "SKIP $m.$tag"; return 0; }
  echo "$(date -Is) BEGIN $m.$tag load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/progress.log"
  env $LEASE PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm PATH=/home/ttuser/tt-bio/env/bin:$PATH \
      "$@" "$PY" -u scripts/perf_regression.py --model "$m" > "$OUT/$m.$tag.log" 2>&1
  echo "$(date -Is) END   $m.$tag rc=$? load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/progress.log"
}

for m in protenix-v2 opendde; do
  arm $m m32 TT_BIO_PROTENIX_TOKEN_BUCKET=1 TT_BIO_PROTENIX_TOKEN_PAD_MULTIPLE=32
  arm $m m64 TT_BIO_PROTENIX_TOKEN_BUCKET=1 TT_BIO_PROTENIX_TOKEN_PAD_MULTIPLE=64
  arm $m off TT_BIO_PROTENIX_TOKEN_BUCKET=0
done
echo "$(date -Is) ALL DONE" >> "$OUT/progress.log"
