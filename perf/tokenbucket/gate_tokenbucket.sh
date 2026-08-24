#!/usr/bin/env bash
# Step G of the token-bucket flip. Ordered by information per device-hour, not by RELEASING.md's
# listing order:
#
#   1. parity        — the correctness gate, the long pole, and the only arm that can block landing
#                      (opendde-abag carries a global_dockq >= 0.50 floor). Every leg here is at an
#                      unaligned token count, so every leg moves. That is the fix working.
#   2. perf          — scoped to the three models that reach the bucketing. Verified by grep: only
#                      tt_bio/protenix.py and tt_bio/opendde.py call bucketed_width /
#                      bucketed_pairformer / _token_bucket, so the other 13 models perf_regression
#                      would sweep cannot be affected and measuring them is wasted device time.
#                      Its input is examples/trpcage.yaml at 20 aa, which the bucket pads 20 -> 64,
#                      so this is the most extreme relative pad anywhere in the gate.
#   3. sizeladder298 — the off-64 diagnostic rung. The standing ladder is 256/512/640/768, every one
#                      a multiple of 64, so it is structurally blind to this lever. 298 is the only
#                      rung that can see it at all. Census only: report it, never re-record off it.
#   4. sizeladder    — the standing arm, last. Expected UNCHANGED, because no default rung is
#                      unaligned. Green here without re-recording is the blindness above, not a pass.
#
#     CARD=<your launch grant> bash perf/tokenbucket/gate_tokenbucket.sh
#
# WT, the worker slug and the gate host are DERIVED, not typed. The first version hardcoded the
# worktree path of the task that wrote it, so a rebase into a differently-named worktree would
# either fail at the cd or, worse, gate a stale tree still sitting at the old path.
set -u
: "${CARD:?set CARD to this launch grant}"
WT=$(cd "$(dirname "$0")/../.." && pwd)
SLUG=${SLUG:-$(basename "$WT")}
HOSTTAG=$(hostname | sed -e s/tt-quietbox2/qb2/ -e s/tt-quietbox/qb1/)
# The gate refuses to score on an interpreter that violates pyproject.toml's declared bounds
# (scripts/gate_guard.py:declared_dependency_problems), and v0.7.0 raised transformers to >=5.5.0
# and huggingface_hub to >=1.5.0. tt-bio-dev/env is still on 4.57.6 / 0.36.2, so the hardcoded
# interpreter this line used to name died in two seconds with "declared version bounds this
# interpreter violates" -- the same hardcoding defect the DockQ resolver below already paid for.
# Ask each candidate whether IT satisfies the tree's own pyproject instead of naming a winner.
if [ -z "${GATE_PYTHON:-}" ]; then
  for c in /home/ttuser/.coworker/rel070/relvenv/bin/python3 /home/ttuser/tt-bio-dev/env/bin/python3 \
           /home/ttuser/tt-bio/env/bin/python3; do
    [ -x "$c" ] || continue
    PYTHONPATH=$WT "$c" -c 'import sys; sys.path.insert(0, sys.argv[1] + "/scripts");
from gate_guard import declared_dependency_problems as d
p = d(sys.argv[1] + "/pyproject.toml")
sys.exit(1 if p else 0)' "$WT" 2>/dev/null || continue
    GATE_PYTHON=$c; break
  done
fi
: "${GATE_PYTHON:?no interpreter on this host satisfies pyproject.toml; set GATE_PYTHON}"
PY=$GATE_PYTHON
cd "$WT" || exit 1
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm
# release_gate shells the DockQ eval out to an interpreter that HAS DockQ installed (the gate venv
# does not). The first version named /home/ttuser/w6_dockq_py, which exists on qb1 only, so on qb2
# the opendde-abag leg folded for 513 s and then died on FileNotFoundError from Popen -- reported as
# "DockQ eval failed", indistinguishable from a real parity failure. Resolve it per host instead.
if [ -z "${OPENDDE_DOCKQ_PYTHON:-}" ]; then
  for c in /home/ttuser/w6_dockq_py /home/ttuser/.abagrank_venv/bin/python3 \
           /home/ttuser/.abag_xm_label_venv/bin/python3; do
    [ -x "$c" ] || continue
    "$c" -c 'import DockQ' 2>/dev/null || continue
    OPENDDE_DOCKQ_PYTHON=$c; break
  done
fi
: "${OPENDDE_DOCKQ_PYTHON:?no interpreter with DockQ found on this host; set OPENDDE_DOCKQ_PYTHON}"
export OPENDDE_DOCKQ_PYTHON
echo "$(date -Is) DockQ interpreter $OPENDDE_DOCKQ_PYTHON"
LEASE="TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:$SLUG"
OUT=perf/tokenbucket/gate
mkdir -p $OUT
HEAD=$(git rev-parse HEAD)
# full_parity_gate fingerprints tt_bio/ + scripts/ (check_workdir_provenance), so those two
# tree hashes are what a green leg is actually valid for. Keying the marker on HEAD instead
# threw a green leg away every time a docs or site commit landed during the run.
CODEKEY=$(git rev-parse HEAD:tt_bio HEAD:scripts | tr -d "\n")
CODEKEY12=$(printf %s "$CODEKEY" | sha256sum | cut -c1-12)
echo "$(date -Is) gating $WT @ $HEAD on $HOSTTAG card $CARD"

if ! timeout 300 env $LEASE PYTHONPATH=$WT "$PY" -u perf/tokenbucket/preflight_card.py \
     >/dev/null 2>&1; then
  echo "PREFLIGHT FAILED on card $CARD. Not gating."; exit 1
fi
echo "$(date -Is) preflight OK on card $CARD, load $(cut -d' ' -f1-3 /proc/loadavg)"

# LEGS selects which legs run (space or comma separated tags, default all). The perf and ladder
# legs are timing legs, the parity leg is not, so when another worker is loading the host the right
# move is to run the correctness pole now and hold the timing legs for a quiet host, rather than let
# a loaded run write a red log that says nothing about this branch.
# A leg needs the granted card to itself, and run() is sequential, so between legs the card MUST be
# free. On 2026-08-24 it was not: a worker child of a fold the gate had already timed out kept
# /dev/tenstorrent/2 for 99 minutes, and the next four legs each burned their full wall clock before
# dying on DeviceInUseError. Both of tt_bio's guards against that need the wedged process's
# interpreter to run (a SIGTERM handler, and a daemon-thread heartbeat), so neither fires while it
# spins inside a ttnn call. Checking here cannot fix that, but it turns four mystery ERRORs into one
# named pid, once.
card_clear() {
  for _ in $(seq 12); do
    holders=$(fuser /dev/tenstorrent/$CARD 2>/dev/null | tr -s ' ')
    [ -z "$holders" ] && return 0
    sleep 5
  done
  echo "CARD $CARD NOT FREE before $1 — held by:$holders"
  ps -o pid,ppid,etime,stat,cmd -p $(echo "$holders" | tr -d 'cefmrw') 2>/dev/null | sed 's/^/    /'
  return 1
}

LEGS=${LEGS:-all}
wanted() { [ "$LEGS" = all ] && return 0; case ",${LEGS//[[:space:]]/,}," in *,$1,*) return 0;; esac; return 1; }

run() {  # rc is the command's, not an echo's (pass 2 logged rc=0 over four hard failures)
  tag=$1; shift
  if ! wanted "$tag"; then echo "HOLD $tag (not in LEGS=$LEGS)"; return 0; fi
  # A .done is only a skip for the commit that wrote it. A rebase across 344 commits touching the
  # same files is exactly the case where "it passed before" is not evidence it still passes.
  if [ "$(cat "$OUT/$tag.done" 2>/dev/null)" = "$CODEKEY" ]; then
    echo "SKIP $tag (green at this HEAD)"; return 0
  fi
  rm -f "$OUT/$tag.done"
  card_clear "$tag" || { echo "=== $(date -Is) SKIP $tag (card busy)"; return 0; }
  echo "=== $(date -Is) BEGIN $tag"
  # keep the previous attempt: a red leg has no .done, so it re-runs on every relaunch and
  # would otherwise overwrite the very output that documents why it is red.
  [ -f "$OUT/$tag.log" ] && mv "$OUT/$tag.log" "$OUT/$tag.$(date +%H%M%S).log"
  env $LEASE PYTHONPATH=$WT ESM_ROOT=$ESM_ROOT OPENDDE_DOCKQ_PYTHON=$OPENDDE_DOCKQ_PYTHON "$@" > "$OUT/$tag.log" 2>&1
  rc=$?
  echo "=== $(date -Is) END $tag rc=$rc"
  [ $rc -eq 0 ] && echo "$CODEKEY" > "$OUT/$tag.done"
  return 0   # never abort the chain on one red leg; every leg's log is wanted
}

# 1. Correctness. Own --workdir, keyed on HEAD: full_parity_gate keys cached per-leg reports on leg
# id alone, so the shared /tmp/full_parity_gate would replay another tree's verdicts as this
# branch's. Its own code fingerprint over tt_bio+scripts then refuses to resume across a rebase,
# which is right, but it leaves the old dir behind, so the dir is named after the same two tree
# hashes it fingerprints. Keyed on HEAD instead, a docs-only commit landing mid-run orphaned a
# 77-minute workdir that the gate would then have refused to reuse for no reason.
run parity "$PY" -u scripts/full_parity_gate.py --workdir /tmp/full_parity_gate-tokenbucket-$CODEKEY12 \
  --workers $HOSTTAG:$CARD \
  --leg protenix-prot-msa --leg protenix-ubq-msa --leg protenix-hsa-msa --leg protenix-9ncy-msa \
  --leg opendde-trpcage-nomsa --leg opendde-prot-prod --leg opendde-abag --leg capacity

# 2. Perf, only where the flip can reach.
for m in protenix-v2 opendde opendde-abag; do
  run "perf-$m" "$PY" -u scripts/perf_regression.py --model $m
done

# 3/4. The ladder: diagnostic rung first, then the standing arm.
export RELEASE_GATE_SIZE_RUNGS=298
run sizeladder298 "$PY" -u scripts/release_gate.py --model size-ladder \
  --size-ladder-models protenix-v2,opendde
unset RELEASE_GATE_SIZE_RUNGS
# Scoped to the two models the flip can reach, same grep-verified reason as the perf leg. The
# unscoped sweep is 6 models x 4 rungs and spent 34 minutes on boltz2 and esmfold2, neither of
# which calls into the bucketing at all. The full arm is the release run's job, not this
# branch's: what this branch owes is evidence that the two models it changes do not regress.
# protenix-v2's rows are provably unchanged (pad 0 at all four rungs); opendde's are the
# informative ones, because its refiner axis 2*n_res-n_GLY is unaligned at every rung.
run sizeladder "$PY" -u scripts/release_gate.py --model size-ladder \
  --size-ladder-models protenix-v2,opendde

echo "=== $(date -Is) ALL LEGS ATTEMPTED"
for f in $OUT/*.log; do
  printf '%-22s %s\n' "$(basename $f .log)" \
    "$([ -f "${f%.log}.done" ] && echo 'rc=0' || echo 'RED or not finished')"
done
