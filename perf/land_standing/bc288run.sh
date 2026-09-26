#!/bin/bash
# EXECUTE the n=288 taped interaction, which is the one thing holding the merge.
#
# The flip opens n=288 on BindCraft 2's taped path: bc288.py showed off=None -> on=(ladder,288,288)
# with 256/320/384 inert. That was argued from bcx-forward's 0.021702 -> 0.018661 and never run.
# tests/test_bindcraft2_hw.py drives a real BC2 gradient step with AF2's 48 Evoformer blocks on
# card, which is exactly that path.
#
# It does NOT need qb1. BindCraft 2 is on qb2 at /home/ttuser/bcx_e2e/bc2 (jax 0.11.2), AF2 params
# resolve to /home/ttuser/.boltz/af2/params with params_model_1_ptm.npz present, and
# bc2/examples/pdl1.json exists -- all three of the test's skip guards are satisfied, checked
# before running. My earlier "cannot run on qb2" was a wrong PATH, not a missing capability.
#
# Both arms: ON is the branch default (the flip), OFF is TT_BIO_TRIATT_DIVIDING_K=0. The firing
# counters come back so "no difference" cannot be confused with "never reached 288".
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/bc288
BC2=/home/ttuser/bcx_e2e/bc2
HOOK=$WT/perf/land_standing/out/widekfire/hook
PY=/home/ttuser/bcx_e2e_venv/bin/python3
CARD=3
mkdir -p "$OUT"
cd "$WT" || exit 1

run () {
  tag=$1
  dk=$2
  dump=$OUT/stats_$tag.jsonl
  rm -f "$dump"
  t0=$(date +%s)
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:land-standing \
      PYTHONPATH="$WT:$BC2:$WT/perf/land_standing" TT_STOCK_STATS_DUMP="$dump" \
      TT_BIO_TRIATT_DIVIDING_K="$dk" \
      "$PY" -m pytest -v -p no:cacheprovider -p ttstats_plugin -s tests/test_bindcraft2_hw.py \
      > "$OUT/log_$tag.txt" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "ARM $tag dividing_k=$dk rc=$rc secs=$((t1 - t0))"
  grep -E "passed|failed|skipped|error" "$OUT/log_$tag.txt" | tail -2
}

run on  1
run off 0
echo BC288_DONE
