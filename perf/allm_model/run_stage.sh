#!/bin/bash
# allm-model: Boltz-2 by stage, on the two trees whose whole-fold ratio is already 1.5006x.
#
# qb2 card 1 (Blackhole p300c). A tree cannot be swapped inside one process, so arms alternate by
# session, old and new in turn, and every session takes benchlock for itself. The clock is held and
# DURING-sampled by the instrument. The instrument file is byte-identical in both trees.
set -u
OLD=/home/ttuser/allm_model/trees/old_boltz2       # 0d69dc1de, the published 23.504 s cell
NEW=/home/ttuser/allm_audit/trees/new              # 47810889f, origin/main pinned 2026-09-21
FIX=$NEW/perf/size512/fixtures                     # one fixture pair for both arms
MSA=/home/ttuser/allm_model/msa_b2_512             # one seeded MSA cache for both arms
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/allm_model/out
BL=/home/ttuser/.coworker/scripts/benchlock.sh
CARD=1
mkdir -p "$OUT" "$MSA"

# Both arms open the chip the same way: the p300c mesh descriptor is resolved off the NEW tree and
# exported to both, as `allm-audit` did.
MGD=$($PY -c "import sys;sys.path.insert(0,'$NEW');from tt_bio.main import _find_ttnn_mesh_graph_descriptor as f;print(f('p150_mesh_graph_descriptor.textproto') or '')" 2>/dev/null)
[ -n "$MGD" ] && export TT_MESH_GRAPH_DESC_PATH="$MGD"

run() {  # run <tree> <commit> <tag> [folds]
  local tree=$1 commit=$2 tag=$3 folds=${4:-cold,A,S,B,S,C}
  if [ -s "$OUT/$tag.json" ] && grep -q '"stages"' "$OUT/$tag.json" 2>/dev/null; then
    echo "=== $(date -u +%H:%M:%SZ) $tag already measured, skipping ==="; return 0
  fi
  echo "=== $(date -u +%H:%M:%SZ) $tag tree=$tree folds=$folds ==="
  # Wait for the box to go quiet BEFORE taking the lock, not inside it. benchlock's own quiet-wait
  # runs with the lock held, so a co-tenant that ignores benchlock (a CPU training job on 2026-09-21)
  # makes one waiter starve every other row for up to 900 s at a time.
  local t0=$SECONDS
  while [ $((SECONDS-t0)) -lt 2400 ]; do
    l=$(cut -d" " -f1 /proc/loadavg)
    awk -v a="$l" 'BEGIN{exit !(a+0<=2.0)}' && break
    echo "$(date -u +%H:%M:%SZ) $tag waiting for a quiet box, loadavg $l (lock NOT held)"
    sleep 30
  done
  BENCHLOCK_WAIT_S=2400 BENCHLOCK_LOAD_WAIT_S=300 "$BL" allm-model -- \
    env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:allm-model \
        ${TT_MESH_GRAPH_DESC_PATH:+TT_MESH_GRAPH_DESC_PATH=$TT_MESH_GRAPH_DESC_PATH} \
        PYTHONPATH="$tree" ALLM_AICLK_DIR="$NEW/perf/c14_bfp8" ALLM_TREE_COMMIT="$commit" \
    "$PY" -u "$tree/perf/allm_model/stagesplit.py" --model boltz2 --size 512 --card $CARD \
      --clock 1350 --fixdir "$FIX" --msadir "$MSA" --folds "$folds" \
      --out "$OUT/$tag.json" --tag "$tag"
  # The chain gates on this status, so it is the measurement command own, not the echo after it.
  local rc=$?
  echo "RC=$rc $tag"
  return $rc
}

old() { run "$OLD" 0d69dc1de "b2stage_old_$1" "${2:-cold,A,S,B,S,C}"; }
new() { run "$NEW" 47810889f "b2stage_new_$1" "${2:-cold,A,S,B,S,C}"; }

# A tape that resolved no seam writes a plausible JSON whose stages are all residual, so the smoke
# session is checked before four more are spent on it.
chain() {
  run "$NEW" 47810889f b2stage_smoke "cold,S" || return 1
  "$PY" "$NEW/perf/allm_model/stagesum.py" --check "$OUT/b2stage_smoke.json" || {
    echo "SMOKE CHECK FAILED -- not spending the sessions"; return 1; }
  new s1; old s1; new s2; old s2
  echo "=== $(date -u +%H:%M:%SZ) chain done ==="
}

"$@"
