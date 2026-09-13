#!/usr/bin/env bash
# D4 fold A/B. Both arms are the SAME tree, selected by TT_BIO_TRIMUL_MM_TRANSPOSE, so the only
# difference between them is the flag. Arm order is the argument; the caller reverses the lead
# between rounds so a monotone drift on the box cannot masquerade as the effect.
set -u
PY=/home/ttuser/tt-bio-dev/env/bin/python3
WT=/home/ttuser/.coworker/wt/util-op-deletes
ART=/home/ttuser/scratch/uod
R="$1"; shift
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:util-op-deletes
for arm in "$@"; do
  [ "$arm" = base ] && F=0 || F=1
  echo "=== $(date -u +%H:%M:%S) r$R arm=$arm flag=$F load=$(cut -d\  -f1 /proc/loadavg)"
  ( cd "$WT" && PYTHONPATH="$WT" TT_BIO_TRIMUL_MM_TRANSPOSE=$F timeout 900 $PY \
      perf/b2x-baseline-attrib/baseline_attrib.py --phases baseline --reps 2 --size 512 \
      --out "$ART/d4_${arm}_r${R}.json" --cifdir "$ART/d4cif_${arm}_r${R}" ) 2>&1 | \
    grep -E "cold|plain|instr|Error|Traceback|FATAL" | tail -6
done
