#!/bin/bash
# size-ladder at 8bede8dfd, ONE MODEL AT A TIME.
#
# WHY PER MODEL. This arm has now been killed three times without banking anything: 2h48m and 2h on
# 2026-09-19, and again today when qb2 hard-reset at 07:29:28Z roughly 2h49m into the run. The arm
# is nine independent per-model ladders behind a single --model value, so a monolithic run stakes
# ~2h45m of device time on a box that has hard-reset three times in two days. `--size-ladder-models
# <m>` scores exactly one ladder and journals it as `size-ladder:<m>` (tests/
# test_gate_size_ladder_resume_scope.py), so a reset now costs the ladder in flight and nothing else.
#
# WHAT A SUBSET RUN DOES NOT CHECK, so it is checked separately rather than lost:
# _size_ladder_coverage_gap(), the assert that every foldable model is on the ladder or carries a
# written exemption. It is skipped on a subset run by design (release_gate.py: "must not be blocked
# by a gap it is not trying to close"). It costs no device time and is asserted in
# check_coverage_gap.py beside this.
#
# WHY 8bede8dfd RATHER THAN bb306f593, where the other twelve arms are banked. main gained
# a8c3fe086 (MSA row-chunk alias fix) after the candidate was cut, and
# `git diff bb306f593..8bede8dfd -- tt_bio` is 12 lines of tt_bio/tenstorrent.py. size-ladder scores
# RUNTIME against scaling exponents, so it gets the tree that actually lands, not the one one merge
# behind. 8bede8dfd is also where fold-models is banked, in this same journal.
#
# Order is cheapest-fold-first, so the most records are banked per unit of wall-clock on a box that
# may reset under us. openfold3 (184 s at 512 aa on the fold arm, the dearest of the nine) runs last.
set -u
WT=/home/ttuser/pvx_gateland_wt
JOURNAL=/home/ttuser/pvx_arms/journal_gateland_8bede8dfd.jsonl
KEEP=/home/ttuser/pvx_gateland_outputs
LOG=/home/ttuser/pvx_arms/gateland_sizeladder.log
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
cd "$WT" || exit 1
mkdir -p "$KEEP"
export PYTHONPATH="$WT"
export TT_BIO_AICLK=1350
export TT_VISIBLE_DEVICES=2
export TT_BIO_LEASE_CARDS=2
export TT_BIO_LEASE_HOLDER=worker:pvx-gate-land

say() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

# AICLK and host contention are sampled DURING the folds by a SEPARATE long-lived process,
# pvx_arms/sample_contention.py -> gateland_sizeladder_contention.jsonl. It is not a child of this
# driver on purpose: the record has to outlive a driver restart, and this driver was restarted once
# already (to set BENCHLOCK_LOAD_WAIT_S). It samples every card, not the first one tt-smi lists --
# three cards here idle at 800 MHz while one folds at 1350, so a first-match grep reports the wrong
# number.

say "=== size-ladder per-model starts, tree $(git rev-parse --short HEAD), card 2, journal $JOURNAL ==="
$PY -c "import tt_bio,sys; sys.stderr.write(\"resolved tt_bio: \"+tt_bio.__file__+chr(10))" 2>&1 | grep resolved >> "$LOG"
$PY -c "import tt_bio.tenstorrent as t; print(\"wide-k default:\", t._SDPA_WIDE_K_DEFAULT, \"_sdpa_wide_k():\", t._sdpa_wide_k())" >> "$LOG" 2>&1

for M in protenix-v1 boltz2 protenix-v2 esmfold2 openbind rf3 opendde nesso1 openfold3; do
  if $PY - "$JOURNAL" "$M" << "PYEOF" >> "$LOG" 2>&1
import json, sys
j, m = sys.argv[1], sys.argv[2]
want = "size-ladder:" + m
latest = None
try:
    for line in open(j):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("arm") == "size-ladder" and want in (r.get("members") or []):
            latest = r
except FileNotFoundError:
    pass
if latest is not None and latest.get("verdict") == "PASS":
    print("[skip] %s already banked PASS at %s" % (want, latest["key"]["commit"]))
    sys.exit(0)
sys.exit(1)
PYEOF
  then continue; fi
  say "--- ladder $M begins"
  # BENCHLOCK_LOAD_WAIT_S is set EXPLICITLY. It defaults to 900 s, after which benchlock warns and
  # PROCEEDS whatever the load is -- and at 07:56Z this box was at loadavg 15.79 on 16 cores, from
  # three of3t bundle_min reference jobs (531 %, 528 %, 115 %) plus pvx-baseline. A timed ladder
  # started there is the contamination this campaign already root-caused; the small rungs carry
  # the size-independent host term, so they inflate most and the exponent goes down. Waiting costs
  # nothing (the card is idle and nobody else wants it) while proceeding costs ~2h45m of device
  # time for a red that cannot be told from a co-tenant. 7200 s keeps it BOUNDED: if the box never
  # goes quiet the arm still runs, under a warning, with gateland_sizeladder_contention.jsonl
  # beside it to say so.
  BENCHLOCK_MAXLOAD=4.0 BENCHLOCK_WAIT_S=28800 BENCHLOCK_LOAD_WAIT_S=7200 \
    bash "$BL" pvx-gate-land -- \
    $PY scripts/release_gate.py --model size-ladder --size-ladder-models "$M" \
        --journal "$JOURNAL" --keep >> "$LOG" 2>&1
  say "--- ladder $M ends rc=$?"
done
say "=== all nine ladders attempted ==="
