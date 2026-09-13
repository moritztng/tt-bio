#!/usr/bin/env bash
# Stack A/B: D1 and D4 together against the tree before either. The two arms are two TREES, not a
# flag, because D1 has no flag -- the lazy mask memo is unconditional -- so the base arm is the
# detached fee8c08e checkout. Arm order is the argument; the caller reverses the lead between
# rounds so a monotone drift on the box cannot masquerade as the effect.
#
# The per-arm timeout is 1800 s, not 900: under a co-tenant at load 14 a 512 aa fold on card 2 ran
# past 900 and the round was lost after the cold fold. Acquire loadavg is echoed per arm so the
# reader can see what each arm was measured against.
set -u
PY=/home/ttuser/tt-bio-dev/env/bin/python3
ART=/home/ttuser/scratch/uod
BASE=/home/ttuser/scratch/uod/base
PATCHED=/home/ttuser/.coworker/wt/util-op-deletes
R="$1"; shift
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:util-op-deletes
for arm in "$@"; do
  [ "$arm" = base ] && T=$BASE || T=$PATCHED
  echo "=== $(date -u +%H:%M:%S) r$R arm=$arm load=$(cut -d\  -f1-3 /proc/loadavg)"
  ( cd "$T" && PYTHONPATH="$T" timeout 1800 $PY \
      perf/b2x-baseline-attrib/baseline_attrib.py --phases baseline --reps 2 --size 512 \
      --out "$ART/stack_${arm}_r${R}.json" --cifdir "$ART/stackcif_${arm}_r${R}" ) 2>&1 | \
    grep -E "cold|plain|instr|Error|Traceback|FATAL" | tail -6
done
