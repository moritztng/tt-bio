#!/bin/bash
# Determinism control: the SAME size folded twice, each in its own fresh process.
# If these two differ, the model is nondeterministic at that shape and the
# batch-vs-solo comparison says nothing about state.
set -u
ROOT=/tmp/whcorr_multishape
REPO=/home/moritz/.coworker/wt/japanfold-wh-correctness-close
PY=/home/moritz/tt-bio/env/bin/python3
cd "$REPO" || exit 1
for t in 02_cdk2_256 03_cdk2_384; do
  for rep in r1 r2; do
    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:japanfold-wh-correctness-close \
    PYTHONPATH=. "$PY" -m tt_bio.main predict "$ROOT/targets_solo/$t/$t.yaml" \
      --model protenix-v2 --seed 0 --single_sequence \
      --out_dir "$ROOT/out/det_${t}_${rep}" >/dev/null 2>&1
    echo "done $t $rep"
  done
done
echo "=== determinism control ==="
for t in 02_cdk2_256 03_cdk2_384; do
  a=$(find "$ROOT/out/det_${t}_r1" -name "*.cif" | head -1)
  b=$(find "$ROOT/out/det_${t}_r2" -name "*.cif" | head -1)
  ha=$(sha256sum "$a" | cut -c1-16); hb=$(sha256sum "$b" | cut -c1-16)
  v=$( [ "$ha" = "$hb" ] && echo DETERMINISTIC || echo "**NONDETERMINISTIC**" )
  echo "$t  r1=$ha  r2=$hb  $v"
done
