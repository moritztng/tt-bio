#!/bin/bash
# One fold arm for the fused-SDPA ragged-pad A/B. Usage:
#   foldab.sh <card> <model> <fixture> <arm-name> <outroot> [extra env assignments...]
# Every arm is its own process: one device context per process, and the compute-config /
# ragged-pad flags are read at import time.
set -euo pipefail
CARD="$1"; MODEL="$2"; FIX="$3"; ARM="$4"; ROOT="$5"; shift 5
WT="$(cd "$(dirname "$0")/../.." && pwd)"
P=/home/ttuser/tt-bio-dev/env/bin/python3
OUT="$ROOT/$ARM"
mkdir -p "$OUT"
cd "$WT"
env PYTHONPATH="$WT" \
    TT_VISIBLE_DEVICES="$CARD" \
    TT_BIO_LEASE_CARDS="$CARD" \
    TT_BIO_LEASE_HOLDER=worker:fused-sdpa-fold-level-root-cause \
    "$@" \
    "$P" -m tt_bio.main predict "$FIX" --model "$MODEL" \
        --out_dir "$OUT" --override --seed 0 \
        --sampling_steps 20 --diffusion_samples 1 \
        --msa_dir /home/ttuser/k6_msa \
    > "$OUT/fold.log" 2>&1
echo "arm $ARM done -> $OUT"
