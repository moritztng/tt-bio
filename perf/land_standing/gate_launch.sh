#!/usr/bin/env bash
# Run the accuracy release gate against THIS worktree's tree, not the shared checkout's.
# release_gate.py imports tt_bio from sys.path, and the shared /home/ttuser/tt-bio-dev is
# ahead on sys.path by default, so without PYTHONPATH the gate scores a tree 15 commits
# behind and says so in a warning that is easy to scroll past.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/gate_m18
mkdir -p "$OUT"
cd "$WT" || exit 1
export PYTHONPATH="$WT"
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:land-standing
# The of3t rows keep this box at loadavg 6-9; the gate's arms are all accuracy (PARSE,
# RMSD/TM, GEOMETRY) so load changes how long a fold takes and not what it scores, but a
# fold timeout IS wall-clock, so raise it rather than let contention read as a red arm.
export RELEASE_GATE_FOLD_TIMEOUT=5400
exec /home/ttuser/tt-bio-dev/env/bin/python3 scripts/release_gate.py \
  --model openfold3 --model opendde --keep \
  --journal "$OUT/journal.json" > "$OUT/gate.log" 2>&1
