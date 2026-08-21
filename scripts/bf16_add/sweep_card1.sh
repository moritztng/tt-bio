#!/bin/bash
# Card 1: protenix-prot-msa arm1 (the arm0 partner, 2.7% slack leg), then the opendde legs
# now that fetch_parity_fixtures.sh restored their reference cifs.
cd "$(dirname "$0")/../.." || exit 1
WT=$PWD
OUT=$WT/scripts/bf16_add/artifacts
P=/home/ttuser/tt-bio-dev/env/bin/python3
export PYTHONPATH=$WT
export TT_VISIBLE_DEVICES=1
export TT_LOGGER_LEVEL=FATAL
export TT_BIO_LEASE_HOLDER=worker:ttnn-add-bf16-rounding-blast-radius

run_leg () {
  leg=$1; arm=$2; wd=$3
  echo "########## leg=$leg arm=$arm $(date -u +%H:%M:%S)"
  TT_BIO_RNE_ADD=$arm "$P" scripts/full_parity_gate.py --workers tt-quietbox:1 \
      --leg "$leg" --out "$OUT/${leg}_arm${arm}.json" --workdir "$wd" 2>&1 | tail -14
  echo "########## done leg=$leg arm=$arm $(date -u +%H:%M:%S)"
}

# Reuse arm0's staged MSA verbatim so the two arms differ only in device arithmetic.
mkdir -p /tmp/bf16sw_pxprot_arm1
cp -r /tmp/gate_pxprot_arm0/msa /tmp/bf16sw_pxprot_arm1/ 2>/dev/null
run_leg protenix-prot-msa 1 /tmp/bf16sw_pxprot_arm1

for arm in 0 1; do run_leg opendde-trpcage-nomsa $arm "/tmp/bf16sw_opendde_arm${arm}"; done
echo "ALL DONE CARD1 $(date -u +%H:%M:%S)"
