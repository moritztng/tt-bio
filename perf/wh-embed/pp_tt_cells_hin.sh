#!/usr/bin/env bash
# Re-run, at n=25, the embed rows whose n=7 warm spread exceeded the brief 1.5 pct rule:
# saprot-35m (13.00), saprot-650m (4.68), esmc-300m (1.67).
#
# Re-running at n=7 would not settle these. A row here is 0.11-0.42 s of device work, so the warm
# spread is host dispatch jitter and 1.5 pct is below the floor of what this box can hold at that
# timescale. What a bigger n buys is a median that is precise even though the spread is wide, plus
# enough samples for an interquartile range, which is the statistic that actually describes a
# jittery distribution. Device cost of the whole script is about 25 s.
set -u
WT=/home/ttuser/.coworker/wt/perf-page-tt-cells
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:perf-page-tt-cells
export TT_BIO_LEASE_TIMEOUT=900
export PYTHONPATH="$WT"
export PROBE_ARCH=blackhole-p150a
[ -d /home/ttuser/esm ] && export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_WAIT_S=3600
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
OUT=$WT/perf/wh-embed/results/pp512_n25
SEQ=$WT/scripts/gpu_vs_tt/fixtures/prot512.seq
SHA=141f7d4730ccf17e116016edc4aceee502d8c9769301ece4d1b64beb496ebf8d
export PY OUT SEQ SHA
mkdir -p "$OUT"

hin_all() {
  while read -r m n b; do
    [ -n "$m" ] || continue
    [ -s "$OUT/$m.json" ] && { echo "SKIP $m"; continue; }
    for try in 1 2 3; do
      echo "=== n25 $m try=$try $(date -u +%H:%M:%SZ)"
      "$PY" -u perf/wh-embed/embed_split_screen.py --model "$m" \
        --n-seqs "$n" --batch-size 8 --assert-batch "$b" \
        --seq-file "$SEQ" --expect-sha "$SHA" --warmup 2 --repeat 25 \
        --out "$OUT/$m.json"
      rc=$?; echo "EXIT_$m=$rc try=$try"
      [ "$rc" -eq 0 ] && break
      [ -s "$OUT/$m.json" ] && break
    done
  done <<ROWS
saprot-35m 7 7
esmc-300m 7 7
saprot-650m 7 7
ROWS
}

for attempt in 1 2 3; do
  echo "### n25 group, benchlock attempt $attempt $(date -u +%H:%M:%SZ)"
  "$BL" perf-page-tt-cells -- bash -uc "$(declare -f hin_all); hin_all"
  rc=$?; echo "N25_GROUP_RC=$rc"
  [ "$rc" -eq 75 ] || break
done
echo "N25_ALL_DONE $(date -u +%H:%M:%SZ)"
