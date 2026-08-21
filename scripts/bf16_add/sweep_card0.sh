#!/bin/bash
# A/B the gate legs that answer "does ttnn.add bf16 tie-breaking move this model's
# committed accuracy number". Cheapest model-covering leg first, one card, both arms.
cd "$(dirname "$0")/../.." || exit 1
WT=$PWD
OUT=$WT/scripts/bf16_add/artifacts
P=/home/ttuser/tt-bio-dev/env/bin/python3
export PYTHONPATH=$WT
export TT_VISIBLE_DEVICES=0
export TT_LOGGER_LEVEL=FATAL
export TT_BIO_LEASE_HOLDER=worker:ttnn-add-bf16-rounding-blast-radius
export ESM_ROOT=/home/ttuser/tt-research/esm

run_leg () {
  leg=$1; arm=$2; tag=${3:-arm$2}
  echo "########## leg=$leg $tag $(date -u +%H:%M:%S)"
  TT_BIO_RNE_ADD=$arm "$P" scripts/full_parity_gate.py --workers tt-quietbox:0 \
      --leg "$leg" --out "$OUT/${leg}_${tag}.json" \
      --workdir "/tmp/bf16sw_${leg}_${tag}" 2>&1 | tail -14
  echo "########## done leg=$leg $tag $(date -u +%H:%M:%S)"
}

for leg in opendde-trpcage-nomsa openfold3-ubq-msa esmfold2-trpcage; do
  run_leg $leg 0
  run_leg $leg 1
done
# A/A control on the leg whose numerator moved most, to separate rounding from diffusion chaos
run_leg boltz2-trpcage-nomsa 0 arm0b
echo "ALL DONE $(date -u +%H:%M:%S)"
