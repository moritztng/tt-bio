#!/bin/bash
# r4t-ceiling step 2: real training step (trunk_step, PERF protocol) under the device profiler
# (Tracy build tt-metal-b2z = ttnn 0.68.0, Python 3.12 tt-bio-dev/env). Phase starts are Tracy signposts.
#   prof_step.sh <out_dir> [card=2] [steps=3] [cand_levers=serving] [keep_book=0]
set -uo pipefail
wt=$(cd "$(dirname "$0")/../.." && pwd)
out=$1; C=${2:-2}; STEPS=${3:-3}; CL=${4:-serving}; KEEP=${5:-0}
M=/home/ttuser/tt-metal-b2z
PY=/home/ttuser/tt-bio-dev/env/bin
raw=/dev/shm/r4t-ceiling-prof-$$
mkdir -p "$out"; cd "$wt"
echo "WAIT $(date -u +%FT%TZ) card=$C steps=$STEPS cand_levers=$CL" > "$out/run.log"
for i in $(seq 1 360); do
  act=$(python3 ~/spd/feed/book.py active $C 2>/dev/null || true)
  if echo "$act" | grep -q fleet-r4t-ceiling && ! echo "$act" | grep -q "release:" && ! fuser /dev/tenstorrent/$C >/dev/null 2>&1; then
    break
  fi
  sleep 15
done
[ "$i" = 360 ] && { echo "NO WINDOW $(date -u +%FT%TZ)" >> "$out/run.log"; exit 4; }
exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { echo "FLOCK TIMEOUT $(date -u +%FT%TZ)" >> "$out/run.log"; exit 3; }
while fuser /dev/tenstorrent/$C >/dev/null 2>&1; do sleep 5; done

( while :; do echo "$(date -u +%T) AICLK $(cat /sys/class/tenstorrent/tenstorrent!$C/tt_aiclk 2>/dev/null)"; sleep 2; done ) > "$out/aiclk.log" 2>&1 &
smp=$!
unset LD_LIBRARY_PATH TT_BIO_LEVERS TT_BIO_LNBW_WEIGHTS TT_BIO_GATE_BW_FUSED TT_BIO_TAPED_KERNELS
export TT_METAL_HOME=$M PYTHONPATH=$wt:$wt/perf:$M/ttnn:$M/tools:$M TT_METAL_PROFILER_DIR=$raw R4T_SIGNPOST=1
export PATH=$PY:$PATH TT_BIO_SDPA_OWN_FORWARD=1 R4T_REBUILD_INLINE=1 OMP_NUM_THREADS=2
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-ceiling
export TT_BIO_SHA=$(git rev-parse --short HEAD)
echo "START $(date -u +%FT%TZ) sha=$TT_BIO_SHA card=$C cl=$CL load=$(cut -d" " -f1 /proc/loadavg)" >> "$out/run.log"
timeout 2400 python3 -m tracy -r --no-op-info-cache -o "$raw" --op-support-count 120000 -- \
  perf/r4t_step/trunk_step.py --n 256 --k 4 --depth 2 --steps "$STEPS" --batch "$HOME/r4t/step/batch00.npz" \
  --exempt perf/r4t_step/g2_exempt.tsv --win-pair card --cand-levers "$CL" --out "$out/ck" > "$out/step.log" 2>&1
rc=$?
echo "PROF EXIT $rc $(date -u +%FT%TZ) load=$(cut -d" " -f1 /proc/loadavg)" >> "$out/run.log"
kill $smp 2>/dev/null || true
csv=$(ls -t "$raw"/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1 || true)
if [ -n "$csv" ]; then
  cp "$csv" "$out/ops.csv"
  echo "CSV rows=$(wc -l < "$out/ops.csv") signposts=$(grep -c ",signpost," "$out/ops.csv" || true)" >> "$out/run.log"
else
  echo "NO CSV FOUND in $raw" >> "$out/run.log"
fi
rm -rf "$raw" "$out/ck"
if [ "$KEEP" != "1" ]; then
  python3 ~/spd/feed/book.py rm fleet-r4t-ceiling >> "$out/run.log" 2>&1 || true
fi
echo "END $(date -u +%FT%TZ)" >> "$out/run.log"
