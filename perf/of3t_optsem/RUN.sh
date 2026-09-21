#!/bin/sh
# Reproduces every number in `~/.coworker/state/of3t-optsem.md`. CPU only, no card.
#
# Two interpreters, because neither stack is installed in the other's environment and
# neither should be: upstream's arm needs pytorch_lightning + torchmetrics, ours needs ttnn.
set -e
REF=${REF:-/home/moritz/.coworker/scratch/of3t-reference/upstream}   # OpenFold3 0.4.3, tag 0.4.3
UP=${UP:-/home/moritz/of3-upstream-venv/bin/python}                  # torch 2.13.0+cpu, pl 2.6.5
OURS=${OURS:-/home/moritz/tt-bio/env/bin/python}                     # torch 2.8.0+cpu, ttnn
S=${S:-/tmp/of3t/optsem}
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../.."
mkdir -p "$S"

# Deliverable 1: does it fire, from upstream's own yaml and pydantic defaults.
$OURS perf/of3t_optsem/fires.py "$REF" "$S/fires.json"

for sc in never single always; do
  $OURS perf/of3t_optsem/make_grads.py $sc "$S/grads_$sc.npz"
  PYTHONPATH=$REF $UP perf/of3t_optsem/upstream_arm.py "$S/grads_$sc.npz" "$S/up_$sc.json"
  $OURS perf/of3t_optsem/ours_arm.py "$S/grads_$sc.npz" "$S/ours_$sc.json"
  $OURS perf/of3t_optsem/compare.py "$S/grads_$sc.npz" "$S/ours_$sc.json" "$S/up_$sc.json" \
        --label "$sc" --out "$S/cmp_$sc.json"
done

# The break control: their optimizer with our skip grafted into it. The reading has to move.
PYTHONPATH=$REF $UP perf/of3t_optsem/upstream_arm.py "$S/grads_single.npz" \
        "$S/upbreak_single.json" --skip-zero-step
$OURS perf/of3t_optsem/compare.py "$S/grads_single.npz" "$S/ours_single.json" \
        "$S/upbreak_single.json" --label "break-control" --out "$S/cmpbreak_single.json"

# `of3t-rebind`'s own D107 harness, unmodified, as an independent second reading.
$OURS perf/of3t_rebind/d107.py "$S/d107.json"
