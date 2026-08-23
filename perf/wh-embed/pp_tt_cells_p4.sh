#!/usr/bin/env bash
# Pass 4. Two things are owed and both are card work.
#
# 1. The three embed rows whose n=7 warm spread broke the brief 1.5 pct rule (saprot-35m 13.00,
#    saprot-650m 4.68, esmc-300m 1.67) get n=25. Each row is 0.11-0.42 s of device work, so the
#    spread is host dispatch jitter, not the model; a bigger n buys a precise median plus an IQR,
#    which is the statistic that describes a jittery distribution. About 25 s of device time.
# 2. esmfold2-fast is re-folded at a load matched to its own control. Pass 3 measured the new row
#    at loadavg 2.15-3.66 and the esmfold2 control at 1.15-1.73, and pass 3 also measured that a
#    co-tenant load is a real multiplicative tax that spread cannot see. A 1.58x claim between two
#    rows measured at different loads is not a claim this page should publish, so it is remeasured.
#    The pass-3 JSON is kept; nothing is overwritten.
#
# Card 1 is this pass grant and its lease read released. Card 0, which carried the pass-3 cells, is
# held by pxdesign-perf-p10 with rfd3-b8-to-4x-p3 queued behind it. Cards are interchangeable here
# and that was measured, not assumed: pass 3 got the identical saprot-650m pooled digest on card 0
# and card 1, and both report the same [11, 10] harvested grid.
set -u
WT=/home/ttuser/.coworker/wt/perf-page-tt-cells
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1
export TT_BIO_LEASE_HOLDER=worker:perf-page-tt-cells
export TT_BIO_LEASE_TIMEOUT=900
export PYTHONPATH="$WT"
export PROBE_ARCH=blackhole-p150a
[ -d /home/ttuser/esm ] && export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_WAIT_S=2100
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
OUT=$WT/perf/wh-embed/results/pp512_n25
FOUT=$WT/perf/wh-embed/results/pp512_loadmatched
SEQ=$WT/scripts/gpu_vs_tt/fixtures/prot512.seq
SHA=141f7d4730ccf17e116016edc4aceee502d8c9769301ece4d1b64beb496ebf8d
export PY OUT FOUT SEQ SHA
mkdir -p "$OUT" "$FOUT"

p4_all() {
  while read -r m n b; do
    [ -n "$m" ] || continue
    [ -s "$OUT/$m.json" ] && { echo "SKIP $m"; continue; }
    for try in 1 2 3; do
      echo "=== n25 $m try=$try $(date -u +%H:%M:%SZ) load=$(cut -d\  -f1 /proc/loadavg)"
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
  if [ ! -s "$FOUT/fold_esmfold2-fast.json" ]; then
    for try in 1 2; do
      echo "=== fold esmfold2-fast try=$try $(date -u +%H:%M:%SZ) load=$(cut -d\  -f1 /proc/loadavg)"
      "$PY" -u perf/esm3p4land/fold_ab.py --model esmfold2-fast --size 512 \
        --rounds 5 --arms ship --out "$FOUT/fold_esmfold2-fast.json"
      rc=$?; echo "EXIT_fold_esmfold2-fast=$rc try=$try"
      [ "$rc" -eq 0 ] && break
      [ -s "$FOUT/fold_esmfold2-fast.json" ] && break
    done
  else
    echo "SKIP fold esmfold2-fast"
  fi
}

for attempt in 1 2 3; do
  echo "### p4 group, benchlock attempt $attempt $(date -u +%H:%M:%SZ)"
  "$BL" perf-page-tt-cells -- bash -uc "$(declare -f p4_all); p4_all"
  rc=$?; echo "P4_GROUP_RC=$rc"
  [ "$rc" -eq 75 ] || break
done
echo "P4_ALL_DONE $(date -u +%H:%M:%SZ)"
