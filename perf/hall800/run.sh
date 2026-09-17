#!/usr/bin/env bash
# One benchlocked device session for the 800 aa Protenix-v2 capacity question.
#
#   perf/hall800/run.sh <run-name> <reps> [--cards 0[,1]]
#
# Holds AICLK at 1350 MHz on every card it uses and samples it the whole time, folds the
# 797-residue two-chain fixture <reps> times in ONE process so reps 2..N are warm, then
# writes runs/<run-name>/{results.json,clock.jsonl,summary.json}. runtime_s comes from the
# fold's own results.json rows, which start after the model is loaded.
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=${1:?run name}; REPS=${2:?reps}; shift 2
CARDS=0
while [ $# -gt 0 ]; do case $1 in --cards) CARDS=$2; shift 2;; *) echo "bad arg $1">&2; exit 64;; esac; done

PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=perf/hall800/runs/$RUN
rm -rf "$D"; mkdir -p "$D"
IN=$D/in; mkdir -p "$IN"
for i in $(seq 1 "$REPS"); do cp perf/hall800/reps/h800_r$i.yaml "$IN/"; done

CLKPIDS=()
for n in ${CARDS//,/ }; do
  python3 perf/hall800/clk.py --node "$n" --target 1350 --out "$D/clock_n$n.jsonl" \
      > "$D/clk_n$n.log" 2>&1 &
  CLKPIDS+=($!)
done
sleep 2
trap 'for p in "${CLKPIDS[@]}"; do kill -TERM $p 2>/dev/null; done' EXIT

NC=$(echo "$CARDS" | tr ',' '\n' | wc -l)
export TT_VISIBLE_DEVICES=$CARDS TT_BIO_LEASE_CARDS=$CARDS TT_BIO_LEASE_HOLDER=worker:hall-capacity-800aa
echo "== $RUN cards=$CARDS reps=$REPS start $(date -Is)" | tee "$D/run.log"
t0=$(date +%s)
timeout 3000 $PY -m tt_bio.main predict "$IN" \
    --model protenix-v2 --fast --diffusion_samples 1 --max_parallel_samples 1 \
    --seed 42 --msa_dir perf/hall800/msa --msa_cache_only \
    --devices "$CARDS" --out_dir "$D/out" --override --debug --log \
    >> "$D/run.log" 2>&1
rc=$?
t1=$(date +%s)
echo "== rc=$rc wall=$((t1-t0))s end $(date -Is)" | tee -a "$D/run.log"

for p in "${CLKPIDS[@]}"; do kill -TERM $p 2>/dev/null; done
sleep 1
python3 perf/hall800/summarise.py "$D" --cards "$CARDS" --reps "$REPS" --wall $((t1-t0)) --rc $rc
exit $rc
