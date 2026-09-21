#!/usr/bin/env bash
# of3t-softgrad: the five arms, one after another on the dispatched card, each with its own
# log so `cost.py` can read the per-structure seconds and the AICLK window back out.
#
# Order is deliberate: the two CONTROLS first. If `shipped` does not reproduce 7.426217 and
# `sm64` does not reproduce 0.0777758 the harness is wrong and the two levers in between mean
# nothing, so they are the arms that run before any lever is paid for.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
L=/home/ttuser/of3t_softgrad
mkdir -p "$L"
for arm in "$@"; do
  echo "CHAIN_ARM_START $arm $(date -u +%FT%TZ)"
  perf/of3t_softgrad/devgrad_sg.sh "$arm" > "$L/devgrad_$arm.log" 2>&1
  rc=$?
  grep -E "ARM_END|AICLK during|median_rel|forward_rel_median|softmax_calls_intercepted|FAILED" \
      "$L/devgrad_$arm.log" | head -8
  echo "CHAIN_ARM_DONE $arm rc=$rc $(date -u +%FT%TZ)"
done
echo "CHAIN_ALLDONE $(date -u +%FT%TZ)"
