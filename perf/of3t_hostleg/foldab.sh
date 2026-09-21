#!/usr/bin/env bash
# of3t-hostleg: one OpenFold3 fold, one arm, with the card's AICLK sampled DURING the fold.
#   foldab.sh <off|on> <seed> <card> <tag>
# `off` is the shipped host legs; `on` sets TT_BIO_OF3_DEVICE_REFATOM=1. Nothing else differs.
# Single-sequence so the run needs no MSA cache and no network, and is the same input every arm.
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ARM=$1; SEED=$2; CARD=$3; TAG=$4
OUT=/tmp/of3t/of3t-hostleg/fold/${TAG}
mkdir -p "$OUT"
case "$ARM" in
  off) export TT_BIO_OF3_DEVICE_REFATOM=0 ;;
  on)  export TT_BIO_OF3_DEVICE_REFATOM=1 ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-hostleg
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cd "$W"

CLK="$OUT/aiclk.tsv"
: > "$CLK"
# AICLK from the kernel driver's sysfs node, not tt-smi: tt-smi's snapshot returns nothing for a
# card whose device is already open, which is every card this samples during a fold (18 samples,
# every value empty, on the first attempt). sysfs answers while the chip is in use and answers
# with an integer.
AICLK_NODE="/sys/class/tenstorrent/tenstorrent!${CARD}/tt_aiclk"
(
  while true; do
    printf '%s\t%s\n' "$(date +%s)" "$(cat "$AICLK_NODE" 2>/dev/null)" >> "$CLK"
    sleep 1
  done
) &
SAMPLER=$!

S=$(date +%s)
echo "=== fold arm=$ARM seed=$SEED card=$CARD tag=$TAG start $(date -u +%FT%TZ) ==="
/home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict examples/ubq.yaml \
  --model openfold3 --out_dir "$OUT" --single_sequence \
  --diffusion_samples 1 --sampling_steps 20 --seed "$SEED" \
  --output_format cif --override > "$OUT/fold.log" 2>&1
rc=$?
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null

echo "ELAPSED ${ARM} $((E-S))s rc=$rc  $(date -u +%FT%TZ)"
awk -F'\t' -v s="$S" -v e="$E" '
  $1+0 >= s && $1+0 <= e && $2 != "" { n++; c=$2+0; t+=c; if (mn=="" || c<mn) mn=c; if (c>mx) mx=c }
  END { if (n==0) print "AICLK: NO SAMPLES IN WINDOW";
        else printf "AICLK during the fold: n=%d mean=%.0f MHz min=%d max=%d\n", n, t/n, mn, mx }
' "$CLK"
find "$OUT" -name '*.cif' -o -name '*.pdb' | sort | while read -r f; do
  printf '%s  %s\n' "$(sha256sum "$f" | cut -c1-16)" "${f#$OUT/}"
done
tail -3 "$OUT/fold.log"
exit $rc
