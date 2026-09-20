#!/bin/bash
# allm-audit: the did-it-transfer cell for the five models pvx-didittransfer did not cover.
# qb2 card 1 (Blackhole p300c, board pair 0/1 -- the partner's clock and watts are recorded per
# fold by the instrument's Sampler, because a busy partner shares this card's power budget).
# Every fold at a pinned 1350 MHz sampled DURING at 4 Hz. Instrument is pvx-baseline's cell.py
# verbatim (md5 21e0770f080a4d965203b59a193107be), so this row's arms, pvx-didittransfer's and
# pvx-baseline's all measure the same region.
#
# A tree cannot be swapped mid-process, so arms alternate by session, old and new in turn.
set -u
T=/home/ttuser/allm_audit/trees
NEW=$T/new                                  # 47810889f -- origin/main, pinned 2026-09-21
FIX=$NEW/perf/size512/fixtures              # one fixture pair for every arm of every model
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/allm_audit/out
BL=/home/ttuser/.coworker/scripts/benchlock.sh
mkdir -p "$OUT"

# The old trees predate nothing relevant here, but the p300c mesh descriptor is resolved off the
# NEW tree and exported to every arm, so both arms of a model open the chip the same way.
MGD=$($PY -c "import sys;sys.path.insert(0,'$NEW');from tt_bio.main import _find_ttnn_mesh_graph_descriptor as f;print(f('p150_mesh_graph_descriptor.textproto') or '')" 2>/dev/null)
[ -n "$MGD" ] && export TT_MESH_GRAPH_DESC_PATH="$MGD"

run() {   # run <tree> <tag> <model> <reps>
  local tree=$1 tag=$2 model=$3 reps=$4
  if [ -s "$OUT/$tag.json" ] && grep -q '"summary"' "$OUT/$tag.json" 2>/dev/null; then
    echo "=== $(date -u +%H:%M:%SZ) $tag already summarised, skipping ==="; return 0
  fi
  echo "=== $(date -u +%H:%M:%SZ) $tag tree=$tree model=$model reps=$reps ==="
  BENCHLOCK_WAIT_S=1200 BENCHLOCK_LOAD_WAIT_S=600 "$BL" allm-audit -- \
    env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:allm-audit \
        ${TT_MESH_GRAPH_DESC_PATH:+TT_MESH_GRAPH_DESC_PATH=$TT_MESH_GRAPH_DESC_PATH} \
        PYTHONPATH="$tree" ALLM_PIN_SOURCE="$NEW/tt_bio/weights.py" \
    "$PY" -u "$tree/perf/allm_audit/pinned_cell.py" --model "$model" --reps "$reps" --clock 1350 \
      --fixdir "$FIX" --out "$OUT/$tag.json" --tag "$tag"
  echo "RC=$? $tag"
}

esm_old() { run "$T/old_esmfold2"  "esm_old_$1"  esmfold2  "${2:-3}"; }
esm_new() { run "$NEW"             "esm_new_$1"  esmfold2  "${2:-3}"; }
odd_old() { run "$T/old_opendde"   "odd_old_$1"  opendde   "${2:-3}"; }
odd_new() { run "$NEW"             "odd_new_$1"  opendde   "${2:-3}"; }
of3_old() { run "$T/old_openfold3" "of3_old_$1"  openfold3 "${2:-3}"; }
of3_new() { run "$NEW"             "of3_new_$1"  openfold3 "${2:-3}"; }

"$@"
