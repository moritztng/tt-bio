#!/bin/bash
# Driver for the 32-chip concurrency-scaling sweep on the JapanFold Wormhole galaxy.
#
# Runs detached (the cloudflared SSH tunnel drops long-lived sessions), writes one JSON
# line per cell so a dropped connection never costs a cell, and keeps every cell at
# identical work (one target, one diffusion sample) so cross-cell comparison is valid.
#
#   E1  concurrency curve 1..32, chips spread across the four PCIe root complexes
#   E2  fixed N=32, all folds packed into M physical cores -- the discriminator between
#       "the host cores are spinning on dispatch back-pressure" and "the host is busy"
#
# Ignore INT/HUP: setsid+nohup alone does not survive the launching session going away on this
# host, and a sweep that dies mid-cell costs the whole cell.
trap "" INT HUP
set -u
SRC=/home/cust-team/mthuening/parity-src
BASE=/home/cust-team/mthuening/g32
YAML=${YAML:-examples/abag_xm/9zen.yaml}

export PYTHONPATH=$SRC HF_HUB_CACHE=/home/cust-team/mthuening/models TT_METAL_LOGGER_LEVEL=FATAL
cd "$SRC" || exit 1
mkdir -p "$BASE"

run() {  # run <phase-args...>
  /usr/bin/python3.10 -u /home/cust-team/mthuening/galaxy_conc_sweep.py \
    --yaml "$YAML" --model opendde-abag \
    --msa-dir /home/cust-team/mthuening/abag_xm/msa_cache \
    --out-root "$BASE/out" --jsonl "$BASE/cells.jsonl" \
    --python /usr/bin/python3.10 --host-threads 2 --diffusion-samples 1 \
    --timeout 900 "$@"
}

echo "=== E1 $(date -Is)"
run --levels "${E1:-1,2,4,8,16,24,28,32}" --chip-order spread
echo "=== E2 $(date -Is)"
run --pin-sweep "${E2:-32,16,8}" --level 32 --chip-order spread
echo "=== SWEEP_DONE $(date -Is)"
