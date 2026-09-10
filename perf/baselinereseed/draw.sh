#!/bin/bash
# One fresh-process draw of perf_regression.py's own per-model measurement
# (2 warmup + 5 timed folds, median). Usage: draw.sh <model> <n> <card>
WT=/home/ttuser/scratch/perf-baseline-reseed-opendde-protenix-v2/wt
S=/home/ttuser/scratch/perf-baseline-reseed-opendde-protenix-v2
m=$1; n=$2; card=${3:-0}
out=$S/logs/${m}_c${card}_d${n}.json
log=$S/logs/${m}_c${card}_d${n}.log
q=$($S/quiet_check.sh) || { echo "SKIP $m draw $n: $q"; exit 1; }
echo "$(date -u +%FT%TZ) model=$m draw=$n card=$card quiet=[$q]" >> $S/draws.log
TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card \
TT_BIO_LEASE_HOLDER=worker:perf-baseline-reseed-opendde-protenix-v2 \
TT_METAL_LOGGER_LEVEL=FATAL LOGURU_LEVEL=WARNING PYTHONPATH="$WT" \
  /home/ttuser/tt-bio-dev/env/bin/python3 "$WT/scripts/perf_regression.py" \
  --measure "$m" --out "$out" > "$log" 2>&1
rc=$?
if [ $rc -ne 0 ] || [ ! -s "$out" ]; then echo "FAIL $m draw $n rc=$rc"; tail -5 "$log"; exit 1; fi
python3 - "$out" "$m" "$n" "$card" <<'PY' >> $S/draws.tsv
import json,sys
d=json.load(open(sys.argv[1]))
print("\t".join([sys.argv[2],sys.argv[3],sys.argv[4],
                 repr(d["throughput"]),repr(d["latency_ms"]),repr(d.get("median_s")),
                 d["unit"],d.get("tt_bio_version","?")]))
PY
tail -1 $S/draws.tsv
