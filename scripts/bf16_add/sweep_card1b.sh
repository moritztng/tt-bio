#!/bin/bash
# Card 1 round 2: OpenDDE. The trpcage reduced fixture's seed dirs are empty even after
# fetch_parity_fixtures.sh (not in the release tarball), so use opendde-prot-prod, whose
# reference cifs are present. Production settings: 10 recycles, 200 sampling steps.
cd "$(dirname "$0")/../.." || exit 1
WT=$PWD
OUT=$WT/scripts/bf16_add/artifacts
P=/home/ttuser/tt-bio-dev/env/bin/python3
export PYTHONPATH=$WT
export TT_VISIBLE_DEVICES=1
export TT_LOGGER_LEVEL=FATAL
export TT_BIO_LEASE_HOLDER=worker:ttnn-add-bf16-rounding-blast-radius

for arm in 0 1; do
  echo "########## leg=opendde-prot-prod arm=$arm $(date -u +%H:%M:%S)"
  TT_BIO_RNE_ADD=$arm "$P" scripts/full_parity_gate.py --workers tt-quietbox:1 \
      --leg opendde-prot-prod --out "$OUT/opendde-prot-prod_arm${arm}.json" \
      --workdir "/tmp/bf16sw_oddeprot_arm${arm}" 2>&1 | tail -14
  echo "########## done leg=opendde-prot-prod arm=$arm $(date -u +%H:%M:%S)"
done
echo "ALL DONE CARD1B $(date -u +%H:%M:%S)"
