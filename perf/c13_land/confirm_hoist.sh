#!/usr/bin/env bash
# Session 2 for the SHIPPED lever, measured alone. The +0.2052 s of record comes from ONE session
# (pass 1, card 0, 2026-09-18), and a fold A/B session is the independent unit: a single session
# cannot see between-session variation, and this campaign exists because measured wins evaporated
# between the screen that found them and the fold that had to pay for them.
#
# Same instrument, second session, different card. Arms base,hoist,base so the hoist fold is
# bracketed by a base fold at each end of its OWN rep -- that gives the paired effect against the
# rep's base mean and the session's own A/A floor from the same five reps, with no extra folds.
set -u
WT=/home/ttuser/.coworker/wt/c13-land-first
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/c13_land
cd "$WT" || exit 1
exec /home/ttuser/.coworker/scripts/benchlock.sh c13-land-first -- env \
  TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:c13-land-first \
  "$PY" perf/c12_compose/fold_compose.py \
    --out "$OUT/confirm_hoist512.json" --cifs "$OUT/confirm_hoist512_cifs" \
    --arms base,hoist,base --reps 5 --size 512 --mhz 1350 --palindrome
