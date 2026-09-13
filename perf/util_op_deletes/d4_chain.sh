#!/usr/bin/env bash
# The whole D4 measurement in one benchlock: two A/B rounds with the lead reversed, then the
# 298 aa control for both arms. One lock, because a lock released between rounds lets a
# co-tenant land between them and the round-to-round A/A floor stops meaning anything.
set -u
ART=/home/ttuser/scratch/uod
WT=/home/ttuser/.coworker/wt/util-op-deletes
PY=/home/ttuser/tt-bio-dev/env/bin/python3
$ART/ab_d4.sh 0 base patched
$ART/ab_d4.sh 1 patched base
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:util-op-deletes
for arm in base patched; do
  [ "$arm" = base ] && F=0 || F=1
  echo "=== $(date -u +%H:%M:%S) 298 control arm=$arm flag=$F load=$(cut -d\  -f1 /proc/loadavg)"
  ( cd "$WT" && PYTHONPATH="$WT" TT_BIO_TRIMUL_MM_TRANSPOSE=$F timeout 900 $PY \
      perf/b2x-baseline-attrib/baseline_attrib.py --phases control --reps 1 --size 512 \
      --out "$ART/d4ctl_$arm.json" --cifdir "$ART/d4cifctl_$arm" ) 2>&1 | \
    grep -E "control|cold|sha|rmsd|Error|Traceback|FATAL" | tail -8
done
echo "=== $(date -u +%H:%M:%S) CHAIN DONE"
