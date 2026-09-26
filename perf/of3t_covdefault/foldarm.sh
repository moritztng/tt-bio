#!/usr/bin/env bash
# of3t-covdefault: one fold, one arm, one model, with the AICLK sampled DURING the fold and a
# runtime CALL CENSUS of the ref-atom leg in every process the run spawns.
#   foldarm.sh <off|on> <seed> <card> <tag> [model]
# Same cell and flags as of3t-hostleg/foldab.sh so the two rows' numbers are comparable; what
# differs is the card (2, this row's grant), the census hook, and that all four clock nodes are
# sampled -- TT_VISIBLE_DEVICES renumbers, so /dev/tenstorrent/<CARD> need not be the chip that
# runs, and a node stuck at 800 MHz through a 10 s fold is the tell.

# One place decides what a valid AICLK is: perf/lib/aiclk.sh, mirroring tt_bio.aiclk.
_L=$(cd "$(dirname "$0")" && pwd); . "${_L%/perf/*}/perf/lib/aiclk.sh" || exit 1
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ARM=$1; SEED=$2; CARD=$3; TAG=$4; MODEL=${5:-openfold3}
OUT=/tmp/of3t/of3t-covdefault/fold/${TAG}
rm -rf "$OUT"; mkdir -p "$OUT"
case "$ARM" in
  off) export TT_BIO_OF3_DEVICE_REFATOM=0 ;;
  on)  export TT_BIO_OF3_DEVICE_REFATOM=1 ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-covdefault
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH=$W/perf/of3t_covdefault${PYTHONPATH:+:$PYTHONPATH}
export OF3T_CENSUS_OUT=$OUT/census
cd "$W"

CLK="$OUT/aiclk.tsv"
: > "$CLK"
( while true; do
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" \
      "$(aiclk 0)" \
      "$(aiclk 1)" \
      "$(aiclk 2)" \
      "$(aiclk 3)" >> "$CLK"
    sleep 1
  done ) &
SAMPLER=$!

S=$(date +%s)
echo "=== fold model=$MODEL arm=$ARM seed=$SEED card=$CARD tag=$TAG start $(date -u +%FT%TZ) ==="
/home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict examples/ubq.yaml \
  --model "$MODEL" --out_dir "$OUT" --single_sequence \
  --diffusion_samples 1 --sampling_steps 20 --seed "$SEED" \
  --output_format cif --override > "$OUT/fold.log" 2>&1
rc=$?
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null

echo "ELAPSED_PROCESS ${ARM} $((E-S))s rc=$rc $(date -u +%FT%TZ)"
python3 "$W/perf/of3t_covdefault/clksum.py" "$OUT" "$S" "$E"
grep -E 'ubq' "$OUT/fold.log" | grep -E '[0-9]+\.[0-9]s' | tail -2
find "$OUT" -name '*.cif' | sort | while read -r f; do
  printf '%s  %s\n' "$(sha256sum "$f" | cut -c1-16)" "${f#$OUT/}"
done
echo "CENSUS:"
python3 "$W/perf/of3t_covdefault/censum.py" "$OUT"
tail -2 "$OUT/fold.log"
exit $rc
