#!/usr/bin/env bash
# Card-3 queue for the L2 landing leg, run after the L4F parity gate releases the card.
#
# Job 1 (new, and it outranks everything else here): the release gate showed L0 and L4F
# disagreeing on prot.yaml -- 117 aa -- while agreeing bit-for-bit on 1ahw_abag. Every
# bit-exactness claim in the state doc was established at 298 aa only. Fold 117 aa on each
# arm: fold_arm.py records intra_run_max_abs_delta_A, so one run settles both whether the
# 117 aa path is deterministic at all and, if it is, which change diverges.
#
# Job 2: E2's eight sites re-measured on qb1 card 3 at ttnn 0.67.4.
set -u
WT=/home/ttuser/.coworker/wt/perfwar-qb1-rebaseline-and-land
OUT=$WT/perf/attn_sites
LOG=$WT/perf/land/out/qb1_next.log
CLOG=$WT/perf/land/out/chain.log
PY=/usr/bin/python3

exec >>"$LOG" 2>&1
echo "=== $(date -u +%H:%M:%S) queued, waiting for card 3"

for _ in $(seq 1 720); do
  grep -q 'CHAIN COMPLETE' "$CLOG" 2>/dev/null && break
  pgrep -f 'land_chain.sh' >/dev/null 2>&1 || break
  sleep 30
done
# /dev/tenstorrent/N is NOT logical card N (UMD ids and device nodes differ), so a
# co-tenant legitimately scheduled on another logical card can hold this node. Wait a
# bounded 10 min for it to clear, then proceed anyway: what section 14 needs from these
# folds is coordinates, and TT_VISIBLE_DEVICES=3 isolates us for correctness even when a
# neighbour perturbs timing.
for _ in $(seq 1 20); do
  fuser /dev/tenstorrent/3 >/dev/null 2>&1 || break
  sleep 30
done
echo "=== $(date -u +%H:%M:%S) card 3 free, starting"

cd "$WT" || exit 1
export PYTHONPATH=$PWD

fold() {  # tag tree commit fused model
  echo "=== $(date -u +%H:%M:%S) fold117 $1 $5"
  TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_HOLDER=worker:perfwar-qb1-rebaseline-and-land \
    timeout 1800 "$PY" perf/land/fold_arm.py --tag "$1" --tree "$2" --expect "$3" \
      --fused "$4" --model "$5" --size 117 --repeat 3
  echo "=== $(date -u +%H:%M:%S) fold117 $1 $5 rc=$?"
}

# protenix-v2 first: it is the model the 298 aa table is built on.
fold L0   arms/L0 834997427f3bddcf9183db14ba215023c0cd3209 0 protenix-v2
fold L2   arms/L2 c42ed26a97709c1a04f139cde30f72df028e8229 0 protenix-v2
fold L3   arms/L3 92c92d9e95d0148173b3cea4633c41bbe8533d93 0 protenix-v2
fold L4   arms/L4 0e9ee663ed9cfc009a30bfbd45ce221e7f55c6a1 0 protenix-v2
fold L4F  arms/L4 0e9ee663ed9cfc009a30bfbd45ce221e7f55c6a1 1 protenix-v2
# and the cross-model confirmation
fold L0o  arms/L0 834997427f3bddcf9183db14ba215023c0cd3209 0 opendde
fold L4o  arms/L4 0e9ee663ed9cfc009a30bfbd45ce221e7f55c6a1 0 opendde

echo "=== $(date -u +%H:%M:%S) 117 aa attribution, protenix-v2"
"$PY" perf/land/compare.py --model protenix-v2 --size 117 --md
echo "=== $(date -u +%H:%M:%S) 117 aa attribution, opendde"
"$PY" perf/land/compare.py --model opendde --size 117 --md --base L0o

# --- Job 2: E2's sites on qb1 card 3 ---
cd "$WT/arms/L0" || exit 1
run() {
  local tag="$1"; shift
  local o="$OUT/rfd3_esm_qb1c3_${tag}.json"
  [ -s "$o" ] && { echo "SKIP $tag"; return 0; }
  echo "=== $(date -u +%H:%M:%S) replay $tag"
  TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_HOLDER=worker:perfwar-qb1-rebaseline-and-land \
    PYTHONPATH="$PWD" timeout 3600 "$PY" -u "$OUT/rfd3_esm_replay.py" \
      --roofs "$WT/perf/land/roofs_card3.json" --out "$o" "$@"
  echo "=== $(date -u +%H:%M:%S) replay $tag rc=$?"
}
run tokens
run atom4032 --n-atom 4032

echo "=== $(date -u +%H:%M:%S) QB1 NEXT COMPLETE"
