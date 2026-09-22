#!/usr/bin/env bash
# of3t-covdefault: THREE folds in ONE process, one arm, so the second and third calls see a warm
# program cache. The served path (`predict_many` on a persistent `_WorkerState`) folds many
# targets per process; the CLI folds one. The flag's cost differs between the two and a
# once-per-process figure cannot say which.
#   foldwarm.sh <off|on> <card> <tag> [model]
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ARM=$1; CARD=$2; TAG=$3; MODEL=${4:-openfold3}
SET=/tmp/of3t/of3t-covdefault/warmset
mkdir -p "$SET"
for n in A B C; do sed "s/^name:.*/name: ubq$n/" "$W/examples/ubq.yaml" > "$SET/ubq$n.yaml" 2>/dev/null \
  || cp "$W/examples/ubq.yaml" "$SET/ubq$n.yaml"; done
OUT=/tmp/of3t/of3t-covdefault/warm/${TAG}
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
CLK="$OUT/aiclk.tsv"; : > "$CLK"
( while true; do
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" \
      "$(cat /sys/class/tenstorrent/tenstorrent!0/tt_aiclk 2>/dev/null)" \
      "$(cat /sys/class/tenstorrent/tenstorrent!1/tt_aiclk 2>/dev/null)" \
      "$(cat /sys/class/tenstorrent/tenstorrent!2/tt_aiclk 2>/dev/null)" \
      "$(cat /sys/class/tenstorrent/tenstorrent!3/tt_aiclk 2>/dev/null)" >> "$CLK"
    sleep 1
  done ) &
SAMPLER=$!
S=$(date +%s)
echo "=== warm model=$MODEL arm=$ARM card=$CARD tag=$TAG start $(date -u +%FT%TZ) ==="
/home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict "$SET" \
  --model "$MODEL" --out_dir "$OUT" --single_sequence \
  --diffusion_samples 1 --sampling_steps 20 --seed 0 \
  --output_format cif --override > "$OUT/fold.log" 2>&1
rc=$?
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
echo "ELAPSED_PROCESS ${ARM} $((E-S))s rc=$rc $(date -u +%FT%TZ)"
python3 "$W/perf/of3t_covdefault/clksum.py" "$OUT" "$S" "$E"
grep -E 'ubq' "$OUT/fold.log" | grep -E '[0-9]+\.[0-9]s' | tail -4
echo "CENSUS:"; python3 "$W/perf/of3t_covdefault/censum.py" "$OUT"
exit $rc
