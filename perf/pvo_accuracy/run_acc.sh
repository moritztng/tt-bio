#!/usr/bin/env bash
# pfm-accuracy arms on the rented A100: kit exact and kit fast over the 11-complex set, same seeds in both
# modes so fast(seed s) vs exact(seed s) isolates the mode and exact(s) vs exact(s') is the seed floor.
# Settings = the customer's: 5 samples, 200 steps, 10 cycles, bf16, MSA from the pc-built a3m files.
# Interleaved so a cut-short session still leaves paired data: exact 101-103, fast 101-103, exact 104-105, fast 104-105.
#   bash run_acc.sh            (expects /root/pfm/acc/{acc.json,msa/}; writes /root/pfm/acc/out/<mode>_<seeds>/)
set -uo pipefail
A=/root/pfm/acc
cd /root/kit/protenix_v2
. venv/bin/activate
export CUDA_HOME=/usr/local/cuda PROTENIX_ROOT_DIR=/weights/protenix
for arm in exact:101,102,103 fast:101,102,103 exact:104,105 fast:104,105; do
  MODE=${arm%%:*}; SEEDS=${arm#*:}; OUT=$A/out/${MODE}_${SEEDS//,/-}
  [ -f "$OUT/rc" ] && grep -q "rc=0" "$OUT/rc" && continue
  mkdir -p "$OUT"
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_event_reasons.active,memory.used \
    --format=csv,noheader -lms 1000 > "$OUT/smi.csv" &
  SMI=$!
  date -u +%FT%T.%NZ > "$OUT/t_start"
  bash run.sh pred --config "${CFG:-a100}" --mode "$MODE" --input $A/acc.json --out_dir "$OUT/pred" --model_name protenix-v2 \
    --seeds "$SEEDS" --cycle 10 --step 200 --sample 5 --dtype bf16 --use_msa true > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  RC=$?
  date -u +%FT%T.%NZ > "$OUT/t_end"
  kill $SMI
  echo "rc=$RC" > "$OUT/rc"
  echo "ARM-DONE $MODE $SEEDS rc=$RC $(find "$OUT/pred" -name '*.cif' | wc -l) cifs"
done
echo ACC-DONE
