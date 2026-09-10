#!/bin/bash
# N fresh-process draws of one gate leg. Best-of-N is the honest estimator for one-sided
# host noise; the median is recorded when a cell is reseeded. Usage: draws.sh <model> <n> [tag]
WT=/home/ttuser/.coworker/wt/perf-excellence-p150a
S=/home/ttuser/.coworker/state/perf-excellence-p150a
m=$1; n=${2:-3}; tag=${3:-draw}
cd "$WT" || exit 1
for i in $(seq 1 "$n"); do
  sleep 15
  q=$("$S/quiet_check.sh")
  out=$(TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:perf-excellence-p150a \
        PYTHONPATH="$WT" ${PY:-/home/ttuser/tt-bio-dev/env/bin/python3} scripts/perf_regression.py --model "$m" 2>&1)
  val=$(echo "$out" | grep -oP "^\[$m\] \K[0-9.]+")
  extra=$(echo "$out" | grep -oP "^\[$m\] .*" | head -1)
  line=$(echo "$out" | grep -E "^$m +[a-z]" | head -1)
  printf "%s\t%s\t%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$tag/$m" "draw$i" "${val:-MEASURE-FAILED}" "$q" >> "$S/draws.tsv"
  { echo "--- $(date -u +%FT%TZ) $tag $m draw$i  quiet: $q"; echo "$extra"; echo "$line"; } >> "$S/perf.log"
done
echo "DRAWS-DONE $tag $m $(date -u +%FT%TZ)" >> "$S/perf.log"
