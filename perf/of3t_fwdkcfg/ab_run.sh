#!/usr/bin/env bash
# The forward softmax compute kernel config, A/B'd on a shipped inference fold with its A/A floor.
#
# Moritz, 2026-09-21: "make sure regular inference is not changed to softmax fp64, not made
# slower. cause it was already in a good state." The lever here is the per-site forward config,
# so the `on` arm sets TT_BIO_SOFTMAX_PRECISE_AB to the token that site resolves under -- never
# `all`, which would also switch the three bf16 sites where the config buys 1.03-2.14x for
# 1.36-2.46x (perf/of3t_fwdkcfg/softmax_cost_foldshapes_qb2c0.json).
#
# Sites, from perf/of3t_fwdkcfg/SITE_DTYPE_CENSUS.json on this fixture:
#   openfold3    openfold3_diffusion_transformer.py:211  144 calls fp32  openfold3.diffusion_transformer
#   protenix-v2  tenstorrent.py:8604                     144 calls fp32  diffusion_transformer.token
#   opendde      only protenix.py:570, bf16, not flipped -- run anyway, the expected answer is
#                that nothing moves, and a control that is expected to be null still has to run
#   boltz2       reaches none of the five sites at all
set -u
WT=/home/ttuser/.coworker/wt/of3t-fwdkcfg
cd "$WT" || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT="$WT/perf/of3t_fwdkcfg"
WORK=$(mktemp -d "${TMPDIR:-/tmp}/of3t-fwdkcfg-ab-XXXXXX")
REPS=${REPS:-2}
CARD=0

export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-fwdkcfg

run() {
  m=$1; token=$2
  echo "=== $m token=$token start $(date -u +%FT%TZ) ==="
  "$PY" perf/of3t_d137tapegate/inference_ab_with_aa_floor.py \
    --model "$m" \
    --fixture "$WT/perf/size512/fixtures/cdk2x2_128.yaml" \
    --base-tree /tmp/of3t-fwdkcfg-base \
    --tree "$WT" \
    --python "$PY" \
    --workdir "$WORK" \
    --card "$CARD" \
    --reps "$REPS" \
    --lever-env TT_BIO_SOFTMAX_PRECISE_AB \
    --lever-value "$token" \
    --tree-must-have softmax_precise_site \
    --base-must-lack "" \
    --out "$OUT/INFERENCE_AB_${m}.json"
  echo "=== $m exit=$? $(date -u +%FT%TZ) ==="
}

run openfold3 openfold3.diffusion_transformer
run protenix-v2 diffusion_transformer.token
run opendde all
echo "=== ALL DONE $(date -u +%FT%TZ) ==="
rm -rf "$WORK"
