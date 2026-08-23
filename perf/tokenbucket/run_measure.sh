#!/usr/bin/env bash
# Sequencer over scripts/gpu_vs_tt/tt_baseline.py --ab-env. Not a benchmark; it only orders the
# runs. It waits for a quiet box BEFORE taking benchlock, so a long wait does not sit on the lock
# and starve other waiters (benchlock is host-scoped and has no fairness).
set -u
WT=/home/ttuser/.coworker/wt/protenix-opendde-token-bucket-flip-measure
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_WAIT_S=7200 BENCHLOCK_LOAD_WAIT_S=1200
mkdir -p perf/tokenbucket

wait_quiet () {   # loadavg under 2.5 and no foreign fold burning CPU, up to 2 h
  local t0=$SECONDS l n
  while [ $((SECONDS-t0)) -lt 7200 ]; do
    l=$(cut -d" " -f1 /proc/loadavg)
    n=$(ps -eo pid,pcpu,args | awk '/python/ && /tt_baseline|tt_bio.main|perf_regression|full_parity_gate|host_cost_probe|boltzgen/ && !/awk|benchlock|run_measure/ && $2+0 > 5 {c++} END{print c+0}')
    if [ "$n" -eq 0 ] && awk -v a="$l" 'BEGIN{exit !(a+0<=2.5)}'; then
      echo "$(date -Is) quiet: load $l"; return 0
    fi
    echo "$(date -Is) busy: load $l, $n foreign fold(s)"
    sleep 30
  done
  echo "$(date -Is) gave up waiting for quiet"; return 1
}

run () {  # run <tag> <model> <fixture-stem> <repeat> <out>
  local tag=$1 model=$2 stem=$3 rep=$4 out=$5
  [ -f "$out" ] && { echo "=== SKIP $tag, $out already exists ==="; return 0; }
  echo "=== $(date -Is) WAIT-QUIET $tag ==="
  wait_quiet || { echo "=== $(date -Is) SKIP $tag, never got a quiet box ==="; return 1; }
  echo "=== $(date -Is) START $tag ==="
  bash /home/ttuser/.coworker/scripts/benchlock.sh protenix-opendde-token-bucket-flip-measure -- \
    env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 \
        TT_BIO_LEASE_HOLDER=worker:protenix-opendde-token-bucket-flip-measure \
        PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm \
    "$PY" -u scripts/gpu_vs_tt/tt_baseline.py --model "$model" --repeat "$rep" \
      --ab-env TT_BIO_PROTENIX_TOKEN_BUCKET --ab-values 1,0 \
      --target perf/size512/fixtures/${stem}.yaml --msa-a3m perf/size512/fixtures/${stem}.a3m \
      --label "$tag" --msa-dir $WT/.msa_tokenbucket \
      --keep-cif perf/tokenbucket/cif_$tag --out "$out"
  echo "=== $(date -Is) END $tag rc=$? ==="
}

run od512 opendde     cdk2x2_512 5 perf/tokenbucket/od512_paired.json
run px512 protenix-v2 cdk2x2_512 5 perf/tokenbucket/px512_paired.json
run od298 opendde     cdk2x2_298 3 perf/tokenbucket/od298_paired.json
run px298 protenix-v2 cdk2x2_298 3 perf/tokenbucket/px298_paired.json
echo "=== $(date -Is) ALL DONE ==="
