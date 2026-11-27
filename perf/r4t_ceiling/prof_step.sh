#!/bin/bash
# r4t-ceiling step 2: the real training step (trunk_step, default levers, PERF protocol) under the device profiler,
# Tracy build of the wheel's ttnn (tt-metal-b2z, v0.68.0). Phase starts are Tracy signposts (R4T_SIGNPOST).
#   prof_step.sh <out_dir> [card] [steps]
# Waits for CARD to be free inside booking fleet-r4t-ceiling, runs once, then drops that booking.
set -uo pipefail
wt=$(cd "$(dirname "$0")/../.." && pwd)
out=$1; C=${2:-0}; STEPS=${3:-3}
M=/home/ttuser/tt-metal-b2z
PY=/home/ttuser/scratch/shipflow/venv-c4f99e695d34/bin
raw=/dev/shm/r4t-ceiling-prof-$$
mkdir -p "$out"; cd "$wt"
echo "WAIT $(date -u +%FT%TZ)" > "$out/run.log"
for i in $(seq 1 360); do
  act=$(python3 ~/spd/feed/book.py active $C 2>/dev/null)
  if echo "$act" | grep -q fleet-r4t-ceiling && ! echo "$act" | grep -q "release:" && [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ]; then break; fi
  sleep 30
done
[ $i = 360 ] && { echo "NO WINDOW $(date -u +%FT%TZ)" >> "$out/run.log"; exit 4; }
( while :; do echo "$(date -u +%T) AICLK $(cat /sys/class/tenstorrent/tenstorrent!$C/tt_aiclk 2>/dev/null)"; sleep 2; done ) > "$out/aiclk.log" 2>&1 &
smp=$!
export TT_METAL_HOME=$M PYTHONPATH=$wt:$wt/perf:$M/ttnn:$M/tools:$M TT_METAL_PROFILER_DIR=$raw R4T_SIGNPOST=1
unset LD_LIBRARY_PATH
export PATH=$PY:$PATH TT_BIO_SDPA_OWN_FORWARD=1
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-ceiling
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD) load $(cut -d" " -f1 /proc/loadavg)" >> "$out/run.log"
timeout 2400 taskset -c 0,1 python3 -m tracy -r --no-op-info-cache -o "$raw" --op-support-count 120000 -- \
  perf/r4t_step/trunk_step.py --n 256 --k 4 --depth 2 --steps $STEPS --batch $HOME/r4t/step/batch00.npz \
  --exempt perf/r4t_step/g2_exempt.tsv --win-pair card --out "$out/ck" > "$out/step.log" 2>&1
echo "PROF EXIT $? $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> "$out/run.log"
kill $smp
csv=$(ls -t "$raw"/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1)
[ -n "$csv" ] && cp "$csv" "$out/ops.csv"
rm -rf "$raw"
python3 ~/spd/feed/book.py rm fleet-r4t-ceiling >> "$out/run.log" 2>&1
echo "END $(date -u +%FT%TZ)" >> "$out/run.log"
