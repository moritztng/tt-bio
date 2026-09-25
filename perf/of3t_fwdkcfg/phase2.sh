#!/usr/bin/env bash
set -u
WT=/home/ttuser/.coworker/wt/of3t-fwdkcfg
cd "$WT" || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT="$WT/perf/of3t_fwdkcfg"
WORK=$(mktemp -d /tmp/of3t-fwdkcfg-p2-XXXXXX)
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-fwdkcfg

echo "=== protenix-v2 A/B, correct token $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_d137tapegate/inference_ab_with_aa_floor.py \
  --model protenix-v2 --fixture "$WT/perf/size512/fixtures/cdk2x2_128.yaml" \
  --base-tree /tmp/of3t-fwdkcfg-base --tree "$WT" --python "$PY" \
  --workdir "$WORK/ab" --card 0 --reps 2 \
  --lever-env TT_BIO_SOFTMAX_PRECISE_AB --lever-value protenix.token_dit \
  --tree-must-have softmax_precise_site --base-must-lack "" \
  --out "$OUT/INFERENCE_AB_protenix-v2.json"
echo "=== protenix-v2 A/B exit=$? $(date -u +%FT%TZ) ==="

echo "=== structure delta $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_fwdkcfg/structure_delta.py \
  --python "$PY" --fixture "$WT/perf/size512/fixtures/cdk2x2_128.yaml" --card 0 \
  --workdir "$WORK/sd" --out "$OUT/STRUCTURE_DELTA_qb2c0.json" \
  --case openfold3=openfold3.diffusion_transformer \
  --case protenix-v2=protenix.token_dit
echo "=== structure delta exit=$? $(date -u +%FT%TZ) ==="
rm -rf "$WORK"
echo "PHASE2 DONE $(date -u +%FT%TZ)"
