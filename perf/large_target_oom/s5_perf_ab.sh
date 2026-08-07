#!/bin/bash
# S5: normal-target perf A/B, branch vs main, interleaved per spec (playbook ACCELERATE rules).
# Runs on qb1 card 3 (BH p150a). tt_baseline.py: model loaded once in-process, one warmup
# fold, then --repeat 5 warm timed folds; we report the full series + median of folds 2-5.
# A/B legs alternate per spec so slow host drift hits both sides alike.
set -u
SRC_B=${SRC_B:-$HOME/oomfix_src}
SRC_A=${SRC_A:-$HOME/oomfix_main}
OUT=${OUT:-$HOME/oomfix_perf}
CARD=${CARD:-3}
ONLY=${ONLY:-all}          # protenix-v2 | opendde | all
mkdir -p "$OUT"
export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_HOLDER=worker:tt-bio-large-target-oom-rootcause
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

# TT_BIO_DRAM_PEAK must stay UNSET here: dram_peak's get_memory_view is not cheap (it
# drains like a sync), and the branch carries denser census tags than main, so timing
# with the probe on fabricates a phantom branch-only regression (measured: p117 44.7s
# with it vs 12.05 without; main itself reads 28.8 with vs ~12 without).
run() { # <src> <model> <target> <a3m> <label> <tag>
  local src=$1; shift
  ( cd "$src" && PYTHONPATH="$src" \
      /usr/bin/python3 scripts/gpu_vs_tt/tt_baseline.py --model "$1" --target "$2" \
      --msa-a3m "scripts/gpu_vs_tt/fixtures/$3" --label "$4" --repeat 5 \
      --out "$OUT/$5.json" ) > "$OUT/$5.log" 2>&1
  echo "$(date -Is) $5 rc=$?" >> "$OUT/progress.txt"
}

# interleaved A/B, branch first then main, per spec
if [ "$ONLY" = all ] || [ "$ONLY" = protenix-v2 ]; then
  run $SRC_B protenix-v2 examples/prot300.yaml prot300.a3m '298 aa' p298_branch
  run $SRC_A protenix-v2 examples/prot300.yaml prot300.a3m '298 aa' p298_main
  run $SRC_B protenix-v2 examples/prot.yaml    prot117.a3m '117 aa' p117_branch
  run $SRC_A protenix-v2 examples/prot.yaml    prot117.a3m '117 aa' p117_main
fi
if [ "$ONLY" = all ] || [ "$ONLY" = opendde ]; then
  run $SRC_B opendde     examples/prot300.yaml prot300.a3m '298 aa' o298_branch
  run $SRC_A opendde     examples/prot300.yaml prot300.a3m '298 aa' o298_main
  run $SRC_B opendde     examples/prot.yaml    prot117.a3m '117 aa' o117_branch
  run $SRC_A opendde     examples/prot.yaml    prot117.a3m '117 aa' o117_main
fi
echo "ALL_PERF_DONE $(date -Is)" >> "$OUT/progress.txt"
