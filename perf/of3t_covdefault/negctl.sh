#!/usr/bin/env bash
# of3t-covdefault: the census NEGATIVE control. A model outside OF3_FAMILY must not reach the
# ref-atom leg, and "it does not import it" is a code reading; this is the run that shows it.
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL=$1; CARD=${2:-2}
OUT=/tmp/of3t/of3t-covdefault/neg/${MODEL}
rm -rf "$OUT"; mkdir -p "$OUT"
export TT_BIO_OF3_DEVICE_REFATOM=1   # ON, so a reach would be maximally visible
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-covdefault
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH=$W/perf/of3t_covdefault${PYTHONPATH:+:$PYTHONPATH}
export OF3T_CENSUS_OUT=$OUT/census
cd "$W"
echo "=== neg model=$MODEL card=$CARD start $(date -u +%FT%TZ) ==="
timeout 1500 /home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict examples/ubq.yaml \
  --model "$MODEL" --out_dir "$OUT" --single_sequence \
  --seed 0 --output_format cif --override > "$OUT/fold.log" 2>&1
echo "rc=$? $(date -u +%FT%TZ)"
grep -E 'ubq' "$OUT/fold.log" | grep -E '[0-9]+\.[0-9]s' | tail -2
echo "CENSUS:"; python3 "$W/perf/of3t_covdefault/censum.py" "$OUT"
tail -2 "$OUT/fold.log"
