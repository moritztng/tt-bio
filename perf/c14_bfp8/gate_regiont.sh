#!/usr/bin/env bash
# Release gate for region T default-ON, precondition 2 of ask 9161.
#
# THE ONE THING THIS DOES THAT EARLIER GATE LAUNCHERS DID NOT. `scripts/release_gate.py` imports
# tt_bio "through the installed dist, NOT by prepending REPO_ROOT" (its own comment at line 205).
# The venv at /home/ttuser/tt-bio-dev/env has tt_bio installed EDITABLE against
# /home/ttuser/tt-bio-dev, so a gate launched from a worktree scores the SHARED CHECKOUT, not the
# branch it was launched in. Measured, not reasoned:
#
#   cd <worktree> && $PY scripts/_whichtt.py   ->  /home/ttuser/tt-bio-dev/tt_bio/__init__.py
#   PYTHONPATH=<worktree> ... same command     ->  <worktree>/tt_bio/__init__.py
#
# and /home/ttuser/tt-bio-dev sits at 480ae2dfe, a release-branch merge, not current main. So the
# PYTHONPATH below is not belt-and-braces, it is the difference between gating this branch and
# gating something else (parity-gate-scores-installed-package-not-checkout).
#
# Grepping tt_bio/tenstorrent.py for the flag, which is what gate4.sh did, CANNOT see this: it
# reads the file on disk beside the script while the gate imports a different file. So the check
# below reads the flag out of the IMPORTED module instead, and refuses on a mismatch.
#
# The size-ladder arm is deliberately NOT in this list. It is the gate's only wall-clock arm, it is
# unguarded and unlocked (no benchlock, no host_quiet, no pair_idle anywhere in release_gate.py),
# and its p300c 256 rung is reps=1 against a +/-0.50 band, which has false-redded four times. Run
# it separately on a quiet box. Everything here is a values, capacity or budget arm and is immune
# to the co-tenancy that the ladder is not.
set -u
WT=/home/ttuser/.coworker/wt/c14-land-tail-regiont
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=${CARD:-3}
cd "$WT" || exit 1
export PYTHONPATH="$WT"

echo "=== REGION T GATE START $(date -Is) commit $(git rev-parse --short HEAD) card $CARD ==="
"$PY" - <<'PYCHECK' || { echo "PREFLIGHT REFUSED"; exit 2; }
import sys, pathlib
wt = pathlib.Path("/home/ttuser/.coworker/wt/c14-land-tail-regiont")
import tt_bio
from tt_bio import tenstorrent as T
pkg = pathlib.Path(tt_bio.__file__).resolve().parent
print("  imported tt_bio   :", pkg)
print("  _TRIATT_B8        :", T._TRIATT_B8)
print("  _B2_DIT_COND_HOIST:", T._B2_DIT_COND_HOIST)
print("  _TRIATT_BIAS_B8   :", T._TRIATT_BIAS_B8)
bad = []
if pkg != (wt / "tt_bio").resolve():
    bad.append("gate would score %s, not this worktree" % pkg)
if T._TRIATT_B8 is not True:
    bad.append("_TRIATT_B8 is %r in the IMPORTED module; this gate exists to score it on" % T._TRIATT_B8)
if T._B2_DIT_COND_HOIST is not True:
    bad.append("_B2_DIT_COND_HOIST is %r; the shipping default is on" % T._B2_DIT_COND_HOIST)
if bad:
    print("REFUSING:"); [print("   -", b) for b in bad]
    sys.exit(1)
print("  preflight OK: the gate will score this branch with region T on")
PYCHECK

for ARM in capacity l1-budget batch-position boltz2 protenix-v2 openfold3 openbind-0 opendde-abag \
           rf3 rf3-1024aa boltzgen rfd3 nesso1 pxdesign esmc-300m esmc-600m; do
  echo
  echo "##### ARM $ARM START $(date -Is) #####"
  TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" TT_BIO_LEASE_HOLDER=worker:c14-land-tail \
    "$PY" scripts/release_gate.py --keep --model "$ARM" 2>&1 \
    | tee "perf/c14_bfp8/gate_regiont_${ARM}.log"
  echo "##### ARM $ARM END rc=${PIPESTATUS[0]} $(date -Is) #####"
done
echo "=== REGION T GATE END $(date -Is) — size-ladder still owed, run it on a quiet box ==="
