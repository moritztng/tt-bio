#!/usr/bin/env bash
# The whole example, start to finish. On a Tenstorrent host: ./run.sh
# On a CPU-only host:                                       ACCEL="--accelerator cpu --no_kernels" ./run.sh
set -euo pipefail
cd "$(dirname "$0")"
ACCEL=${ACCEL:-}
W=${WORK:-work}

python fetch.py "$W" 1UBQ 1PGA 1CSP 1ENH 1SHG 2CI2 3CHY
mkdir -p "$W/train" "$W/test"
mv "$W/inputs/3chy.yaml" "$W/test/"
mv "$W"/inputs/*.yaml "$W/train/"

# 1. fold the training proteins, exporting the pair representation with a head
tt-bio predict "$W/train" --model boltz2 --single_sequence $ACCEL \
    --head contact_head.py:pair_features --out_dir "$W/runs"

# 2. train the contact head with the custom loss, on the host
python train_head.py "$W/runs" "$W/truth"

# 3. fold a protein the head never saw, with the trained head attached
tt-bio predict "$W/test" --model boltz2 --single_sequence $ACCEL \
    --head contact_head.py:ContactHead --out_dir "$W/runs"

# 4. score the head's contacts and the fold's own structure against the deposited structure
python score.py "$W/runs/boltz2_results_test/structures" "$W/truth"
