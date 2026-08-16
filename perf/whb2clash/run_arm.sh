#!/bin/bash
# Fold one target in one (K3, lever C) corner, on either architecture, and leave behind proof
# from inside the folding process that the corner took effect.
#
# The gate this serves compares four corners on identical input, so its one fatal failure mode
# is an arm that never reaches the constant it is supposed to move -- an A/B that is silently an
# A/A. `perf/whb2clash/hook/sitecustomize.py` is on PYTHONPATH here, which is what makes the
# probe land in the *spawned* fold child rather than only in this launcher.
#
# Usage:
#   run_arm.sh <outdir> <yaml> <k3:0|1> <slmc|-> <device> <msa_dir> [diffusion_samples]
set -euo pipefail
OUT=$1; YAML=$2; K3=$3; SLMC=$4; DEV=$5; MSA=$6; SAMPLES=${7:-1}
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV=${WHB2_PY:-/home/ttuser/tt-bio-dev/env/bin/python}
mkdir -p "$OUT/probe"
ENVARGS=()
[ "$SLMC" = "-" ] || ENVARGS+=("WHB2_FORCE_SLMC=$SLMC")
env TT_VISIBLE_DEVICES="$DEV" \
    TT_BIO_LEASE_HOLDER=worker:wh-boltz2-640aa-clash-rootcause \
    TT_BIO_SDPA_DIV_K="$K3" \
    WHB2_PROBE="$OUT/probe" \
    "${ENVARGS[@]}" \
    PYTHONPATH="$WT/perf/whb2clash/hook:$WT" \
    "$VENV" -m tt_bio.main predict "$YAML" \
      --out_dir "$OUT" --model boltz2 --accelerator tenstorrent --debug --log \
      --output_format cif --fast --msa_dir "$MSA" --diffusion_samples "$SAMPLES"
