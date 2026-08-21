#!/bin/bash
# Card 1 round 3: A/A controls for the two load-bearing A/B claims, on the SAME card the
# arms ran on (both boltz2 and opendde arms were card 1; the arm0b queued on card 0 is the
# wrong card to control them).
#   boltz2  : is the 0.0722 -> 0.0456 numerator move the rounding, or seed chaos?
#   opendde : is the D floor 2.121 -> 4.172 blowup the rounding, or run-to-run variation in D?
# The opendde one is the important one: "the fix makes the device 1.8x less self-consistent"
# is the claim the whole OpenDDE verdict rests on.
cd "$(dirname "$0")/../.." || exit 1
WT=$PWD
OUT=$WT/scripts/bf16_add/artifacts
P=/home/ttuser/tt-bio-dev/env/bin/python3
export PYTHONPATH=$WT
export TT_VISIBLE_DEVICES=1
export TT_LOGGER_LEVEL=FATAL
export TT_BIO_LEASE_HOLDER=worker:ttnn-add-bf16-rounding-blast-radius

aa () {
  leg=$1; tag=$2
  echo "########## leg=$leg $tag (arm0 repeat) $(date -u +%H:%M:%S)"
  TT_BIO_RNE_ADD=0 "$P" scripts/full_parity_gate.py --workers tt-quietbox:1 \
      --leg "$leg" --out "$OUT/${leg}_${tag}.json" --workdir "/tmp/bf16sw_${tag}" 2>&1 | tail -12
  echo "########## done leg=$leg $tag $(date -u +%H:%M:%S)"
}

aa boltz2-trpcage-nomsa aa_card1
aa opendde-prot-prod aa_card1
echo "ALL DONE CARD1C $(date -u +%H:%M:%S)"
