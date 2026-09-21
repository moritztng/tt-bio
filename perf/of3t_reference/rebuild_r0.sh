#!/usr/bin/env bash
# Rebuild the r = 0 BUNDLE-MIN gradient on a rented GPU box, twice, and reproduce it against
# itself. This is the whole publication recipe: everything it needs is either fetched from a
# public URL or carried in with a hash.
#
# The batch is NOT rebuilt here and must not be. Featurization is deterministic within one
# environment and environment-dependent across one -- rebuilding the 20 batches twice on the same
# box gave 20/20 byte-identical files and rebuilding them on pc gave 0/20, with the same entry
# order and the same token counts. So the bytes are carried in from the canonical host and the
# hash is checked before use.
set -euo pipefail

ROOT=${ROOT:-/root/of3t}
TAG=v0.5.0
COMMIT=c4771653c5d0a3ebb0b3af71b05efd64bc44ee86
CKPT_URL=https://openfold3-data.s3.amazonaws.com/openfold3-parameters/of3-p2-155k.pt
BATCH_SHA=3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f

mkdir -p "$ROOT" && cd "$ROOT"

echo "=== box ==="
nvidia-smi --query-gpu=name,clocks.max.sm,memory.total --format=csv
lscpu | grep -E '^(Model name|CPU\(s\)):' ; free -g | head -2; cat /proc/loadavg

echo "=== deps ==="
pip install -q --no-input "pytorch-lightning>=2.1" ml-collections biotite "rdkit<2026" \
  pdbeccdutils kalign-python ijson dm-tree einops torchmetrics deepspeed 2>&1 | tail -2 || true

[ -d openfold-3 ] || git clone -q --branch "$TAG" https://github.com/aqlaboratory/openfold-3.git
test "$(git -C openfold-3 rev-parse HEAD)" = "$COMMIT"

[ -s of3-p2-155k.pt ] || curl -sfL -o of3-p2-155k.pt "$CKPT_URL"
echo "checkpoint sha256: $(sha256sum of3-p2-155k.pt | cut -d' ' -f1)"

echo "$BATCH_SHA  batch_step003.pt" | sha256sum -c -

export PYTHONPATH="$ROOT/openfold-3"

# Run A carries the finite-difference validation. Run B is a second, independent production of the
# same gradient in a fresh process: PROTOCOL A13, the check that a reference has been reproduced
# rather than merely measured.
for run in A B; do
  extra=""
  [ "$run" = B ] && extra="--fd-samples 0"
  echo "=== run $run ==="
  # Clock sampled DURING the run, with the co-tenant count, because a figure without its clock is
  # not a measurement and a shared card is an artifact rather than a result.
  ( while true; do
      nvidia-smi --query-gpu=clocks.sm,utilization.gpu --format=csv,noheader
      nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l
      sleep 5
    done > "clock_${run}.txt" ) & SAMPLER=$!
  trap 'kill $SAMPLER 2>/dev/null || true' EXIT
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python bundle_min.py --batch batch_step003.pt --batch-sha256 "$BATCH_SHA" \
      --out "out_$run" --dtype float64 --num-recycles 0 --checkpoint of3-p2-155k.pt \
      --fd-h 1e-4 --fd-samples 16 $extra 2>&1 | tee -a "run_${run}.log" | tail -40
  kill $SAMPLER 2>/dev/null || true
  echo "clock samples: $(grep -c MHz "clock_${run}.txt") range $(grep MHz "clock_${run}.txt" | sort -n | sed -n '1p;$p' | tr '\n' ' ')"
done

echo "=== PROTOCOL A13: is it reproduced? ==="
python compare_grads.py out_A/grads_f64.pt out_B/grads_f64.pt --json-out reproduction_A13.json
