#!/bin/bash
# Is boltz2-prot-nomsas GAP caused by the assembly, or is it what main does too?
#
# parity3 returned PASS / GAP / PASS on trpcage / prot-nomsa / hsa-nomsa. GAP is documented
# gate-failing unless it reproduces a committed GAP-evidenced entry, and the gate still
# printed GATE PASS and exited 0 -- so the verdict has to be decided by evidence, not by
# reading the summary line. Same leg, same card, same references, two trees.
#
# The target is intrinsically noisy: on mains own metrics this leg has a reference-vs-
# reference floor spanning 2.897-6.936 A with cross max 7.492 and cross_over_floor 0.920.
set -u
PY=/home/ttuser/tt-bio/env/bin/python3
for ARM in mainref cutover; do
  case $ARM in
    mainref) T=/home/ttuser/.coworker/wt/japanfold-wh-cutover-mainref ;;
    cutover) T=/home/ttuser/.coworker/wt/japanfold-wh-cutover ;;
  esac
  echo "=== $ARM $(git -C $T rev-parse --short HEAD) start $(date -u +%H:%M:%S)"
  ESM_ROOT=/home/ttuser/tt-research/esm PYTHONPATH="$T" TT_METAL_LOGGER_LEVEL=FATAL \
  TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
    "$PY" "$T/scripts/full_parity_gate.py" --workers tt-quietbox:0 --fresh \
      --workdir /home/ttuser/.coworker/wt/japanfold-wh-cutover/perf/whcut/out/gap_$ARM \
      --leg boltz2-prot-nomsa > /home/ttuser/.coworker/wt/japanfold-wh-cutover/perf/whcut/out/gap_$ARM.log 2>&1
  echo "EXIT $ARM = $?"
  grep -E "boltz2-prot-nomsa" /home/ttuser/.coworker/wt/japanfold-wh-cutover/perf/whcut/out/gap_$ARM.log | tail -1
done
echo "GAP AB DONE $(date -u +%FT%TZ)"
