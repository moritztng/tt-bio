#!/usr/bin/env bash
# The six release-gate arms gate3 never reached. gate3 (unit c13-gate3, 18:15:39Z, commit
# fcf82cfc0) scored EIGHT arms PASS with zero contention and was then killed mid-capacity by the
# host reboot at 19:05:43Z -- standing qb2 watchdog behaviour, the same thing that took launch 1
# at 11:03Z. Re-running the whole 3h36m set to re-earn eight verdicts the log already carries is
# how this row has spent four passes, so this resumes by arm instead.
#
# Each arm is its own driver process, run sequentially, so no arm can become its own co-tenant.
# Cheapest first; the size-ladder (2h53m of the full gate's 3h35m) goes last so a reboot costs
# the least already-earned verdict.
set -u
WT=/home/ttuser/.coworker/wt/c13-land-first
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
echo "=== GATE4 START $(date -Is) commit $(git rev-parse --short HEAD) card 0 ==="
echo "--- the shipping configuration, read out of the running tree ---"
grep -n '^_UNFUSED_SILU\|^_B2_DIT_COND_HOIST' tt_bio/tenstorrent.py
echo "--- the p300c 256 cells this ladder scores against ---"
"$PY" -c 'import json
for m in ("boltz2", "openbind"):
    d = json.load(open("docs/size_ladder_baseline.d/%s.json" % m))
    c = d["cards"]["p300c"]["models"][m]
    print("  %-9s %s commit=%s" % (m, c["exponents"], c.get("commit")))'
for ARM in capacity l1-budget batch-position esmc-300m esmc-600m size-ladder; do
  echo
  echo "##### ARM $ARM START $(date -Is) #####"
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:c13-land-first \
    "$PY" scripts/release_gate.py --keep --model "$ARM" 2>&1 | tee "perf/c13_land/gate4_${ARM}.log"
  echo "##### ARM $ARM END rc=${PIPESTATUS[0]} $(date -Is) #####"
done
echo "=== GATE4 END $(date -Is) ==="
