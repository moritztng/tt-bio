#!/usr/bin/env bash
# 4649 engine-leak check: both boltz-2 levers claim to be boltz-2-exclusive by construction.
# Prove it. Fold a non-boltz-2 model with both flags forced OFF and forced ON and require the
# full CIF sha256 to be byte-identical -- an engine-wide default that silently moves another
# model is not mergeable at any speedup (state/answered/4649-decision.md).
#
# Both arms set the flags EXPLICITLY, so this is immune to the default flip landing mid-sweep.
set -u
WT=${WT:-/home/ttuser/.coworker/wt/b2x-integrate}
cd "$WT" || exit 1
PY=${PY:-/home/ttuser/tt-bio-dev/env/bin/python3}
OUT=${OUT:-$WT/perf/b2x-integrate/leak}
CHIP=${CHIP:-1}
HOLDER=${HOLDER:-worker:b2x-integrate}
mkdir -p "$OUT"
COMMON="--single_sequence --seed 0 --recycling_steps 1 --sampling_steps 20 --diffusion_samples 1 --output_format cif"
for model in "$@"; do
  for arm in off:0 on:1 off2:0; do
    tag=${arm%%:*}; flag=${arm##*:}
    d="$OUT/${model}_${tag}"
    rm -rf "$d"; mkdir -p "$d"
    echo "=== $model arm $tag flags=$flag start $(date -u +%H:%M:%S)"
    TT_VISIBLE_DEVICES="$CHIP" TT_BIO_LEASE_CARDS=0,"$CHIP" TT_BIO_LEASE_HOLDER="$HOLDER" \
    BOLTZ2_TOKEN_DIT_SDPA="$flag" TT_BIO_ATOM_AXIS_BUCKET="$flag" \
    PYTHONPATH="$WT" timeout 2400 "$PY" -m tt_bio.main \
      predict examples/615.yaml --model "$model" $COMMON --out_dir "$d" \
      2>&1 | grep -E "Done:|Error|Traceback|failed|✗" | tail -4
    find "$d" -name "*.cif" -exec sha256sum {} \; | sed "s|$OUT/||" | tee -a "$OUT/shas.txt"
    echo "--- $model $tag end $(date -u +%H:%M:%S)"
  done
done
echo "LEAKCHECK_DONE $(date -u +%H:%M:%S)"
